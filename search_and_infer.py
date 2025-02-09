import argparse
import os
import torch
import numpy as np
import json
from pathlib import Path
import yaml

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# Import model and tokenizer
from models.multimodal_to_smiles import MultiModalToSMILESModel
from models.smiles_tokenizer import SmilesTokenizer


def load_config(config_path=None):
    # Default minimal configuration (can be extended via a YAML file)
    default_config = {
        'model': {
            'max_seq_length': 512,
            'max_nmr_length': 128,
            'max_memory_length': 128,
            'embed_dim': 256,
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1,
            'use_stablemax': False,
            'width_basis': 13
        },
        'data': {
            'tokenized_dir': 'tokenized_baseline/data'
        }
    }
    if config_path is not None:
        with open(config_path, 'r') as f:
            custom_config = yaml.safe_load(f)
            def update_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = update_dict(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d
            update_dict(default_config, custom_config)
    return default_config


def greedy_decode(model, nmr_tokens, ir_data, tokenizer, max_len=128, device=None):
    """
    Greedy decode function adapted from train_autoregressive_mup.py.
    For a single example, we add a batch dimension to nmr_tokens and ir_data if available.
    """
    if device is None:
        device = next(model.parameters()).device
    BOS_TOKEN_ID = tokenizer.cls_token_id
    EOS_TOKEN_ID = tokenizer.sep_token_id
    
    model.eval()
    with torch.no_grad():
        # Prepare batch dimension
        if nmr_tokens is not None and nmr_tokens.dim() == 1:
            nmr_tokens = nmr_tokens.unsqueeze(0)
        if ir_data is not None and ir_data.dim() == 1:
            ir_data = ir_data.unsqueeze(0)
        batch_size = 1
        
        # Start token for decoding
        current_token = torch.tensor([[BOS_TOKEN_ID]], device=device)
        
        # Encode spectral data
        memory = model.encoder(None, ir_data, None)
        if memory is None:
            memory = torch.zeros(batch_size, model.decoder.max_memory_length, model.decoder.memory_dim, device=device)
        
        generated_sequences = [[BOS_TOKEN_ID] for _ in range(batch_size)]
        max_len = min(max_len, model.decoder.max_seq_length)
        finished_sequences = [False] * batch_size
        
        for _ in range(max_len):
            logits = model.decoder(tgt=current_token, memory=memory, nmr_tokens=nmr_tokens)
            next_token = logits[:, -1:].argmax(dim=-1)
            for i in range(batch_size):
                if not finished_sequences[i]:
                    token = next_token[i].item()
                    generated_sequences[i].append(token)
                    if token == EOS_TOKEN_ID:
                        finished_sequences[i] = True
            if all(finished_sequences):
                break
            current_token = torch.cat([current_token, next_token], dim=1)
        
        decoded_sequences = []
        for seq in generated_sequences:
            try:
                eos_idx = seq.index(EOS_TOKEN_ID)
                seq = seq[:eos_idx]
            except ValueError:
                pass
            decoded = tokenizer.decode(seq[1:])  # Skip BOS token
            decoded_sequences.append(decoded)
        return decoded_sequences


def compute_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return fp


def tanimoto_similarity(fp1, fp2):
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


class SimpleSpectralSmilesDataset:
    """
    A simplified dataset loader to read source and target SMILES, and IR data from numpy files.
    This mimics parts of SpectralSmilesDataset from train_autoregressive_mup.py.
    """
    def __init__(self, data_dir, split='test', smiles_tokenizer=None, spectral_tokenizer=None, max_smiles_len=512, max_nmr_len=128):
        self.data_dir = Path(data_dir)
        src_file = self.data_dir / f"src-{split}.txt"
        tgt_file = self.data_dir / f"tgt-{split}.txt"
        with open(src_file) as f:
            self.sources = [line.strip() for line in f]
        with open(tgt_file) as f:
            self.targets = [line.strip().replace(" ", "") for line in f]
        self.smiles_tokenizer = smiles_tokenizer
        self.spectral_tokenizer = spectral_tokenizer
        self.max_smiles_len = max_smiles_len
        self.max_nmr_len = max_nmr_len
        
        # Load IR data if available
        ir_file = self.data_dir / f"ir-{split}.npy"
        self.ir_data = None
        if ir_file.exists():
            try:
                self.ir_data = np.load(ir_file, mmap_mode='r')
                if len(self.ir_data.shape) == 1:
                    num_samples = len(self.sources)
                    feature_dim = self.ir_data.shape[0] // num_samples
                    self.ir_data = self.ir_data.reshape(num_samples, feature_dim)
            except Exception as e:
                print(f"[Dataset] Failed to load IR data: {e}")
                self.ir_data = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # For inference we need target tokens, nmr tokens, and IR data
        target_seq = self.targets[idx]
        target_tokens = self.smiles_tokenizer.encode(
            target_seq,
            add_special_tokens=True,
            max_length=self.max_smiles_len,
            truncation=True
        )
        target_tokens = torch.tensor(target_tokens, dtype=torch.long)

        source_seq = self.sources[idx]
        nmr_tokens = source_seq.split()
        nmr_token_ids = [self.spectral_tokenizer.get(token, self.spectral_tokenizer.get("<UNK>")) for token in nmr_tokens]
        if len(nmr_token_ids) > self.max_nmr_len:
            nmr_token_ids = nmr_token_ids[:self.max_nmr_len]
        nmr_tokens = torch.tensor(nmr_token_ids, dtype=torch.long)

        ir_tensor = None
        if self.ir_data is not None:
            ir_tensor = torch.tensor(self.ir_data[idx].copy(), dtype=torch.float32)

        return target_tokens, (ir_tensor, None), nmr_tokens, None


def main():
    parser = argparse.ArgumentParser(description='Search dataset for nearest SMILES and run inference on spectra')
    parser.add_argument('--query', type=str, help='Query SMILES string (ignored if --use_candidates is set)')
    parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to find')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pretrained model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (YAML)')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (train/val/test)')
    parser.add_argument('--use_candidates', action='store_true', help='If set, use hardcoded candidate SMILES instead of --query')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load tokenizers
    current_dir = os.path.dirname(os.path.realpath(__file__))
    vocab_path = os.path.join(current_dir, 'vocab.txt')
    tokenizer = SmilesTokenizer(vocab_file=vocab_path)

    nmr_vocab_path = Path(config['data']['tokenized_dir']).parent / 'vocab.json'
    if not nmr_vocab_path.exists():
        raise FileNotFoundError(f"NMR vocabulary not found at {nmr_vocab_path}")
    with open(nmr_vocab_path) as f:
        nmr_tokenizer = json.load(f)

    # Create dataset
    dataset = SimpleSpectralSmilesDataset(
        data_dir=config['data']['tokenized_dir'],
        split=args.split,
        smiles_tokenizer=tokenizer,
        spectral_tokenizer=nmr_tokenizer,
        max_smiles_len=config['model']['max_seq_length'],
        max_nmr_len=config['model']['max_nmr_length']
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smiles_vocab_size = len(tokenizer)
    token_ids = list(nmr_tokenizer.values())
    nmr_vocab_size = max(token_ids) + 1

    model = MultiModalToSMILESModel(
        smiles_vocab_size=smiles_vocab_size,
        nmr_vocab_size=nmr_vocab_size,
        max_seq_length=config['model']['max_seq_length'],
        max_nmr_length=config['model']['max_nmr_length'],
        max_memory_length=config['model']['max_memory_length'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        verbose=False,
        use_stablemax=config['model'].get('use_stablemax', False)
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if args.use_candidates:
        candidates = [
            ("atp", "C1=NC2=C(C(=N1)N)N=CN2C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O"),
            ("gtp", "C1=NC2=C(N1C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)NC(=NC2=O)N"),
            ("uracil", "O=C1C=CNC(=O)N1"),
            ("citosine", "C1=C(NC(=O)N=C1)N"),
            ("guanine", "C1=NC2=C(N1)C(=O)N=C(N2)N"),
            ("adenine", "C1=NC2=C(N1)C(=NC=N2)N")
        ]

        for name, smi in candidates:
            query_fp = compute_fingerprint(smi)
            if query_fp is None:
                print(f"Invalid candidate SMILES for {name}. Skipping.")
                continue
            similarities = []
            for i, smiles in enumerate(dataset.targets):
                fp = compute_fingerprint(smiles)
                sim = tanimoto_similarity(query_fp, fp)
                similarities.append((i, sim))
            similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
            top_k = similarities[:args.k]
            print(f"\nCandidate: {name}, SMILES: {smi}")
            print(f"Top {args.k} entries similar to candidate {name}:")
            for idx, sim in top_k:
                print(f"Index: {idx}, SMILES: {dataset.targets[idx]}, Tanimoto: {sim:.4f}")
            print(f"\nRunning inference for candidate {name}:")
            for idx, sim in top_k:
                target_tokens, (ir_data, _), nmr_tokens, _ = dataset[idx]
                if ir_data is not None:
                    ir_data = ir_data.to(device)
                nmr_tokens = nmr_tokens.to(device)
                prediction = greedy_decode(model, nmr_tokens, ir_data, tokenizer, device=device)
                print(f"Index: {idx}")
                print(f"Dataset SMILES: {dataset.targets[idx]}")
                print(f"Tanimoto similarity: {sim:.4f}")
                print(f"Predicted SMILES: {prediction[0]}")
                print("-"*40)
        return
    else:
        # Process using provided --query
        if not args.query:
            print("Error: Either provide a --query SMILES or use --use_candidates flag.")
            return

        query_fp = compute_fingerprint(args.query)
        if query_fp is None:
            print('Invalid query SMILES.')
            return

        similarities = []
        for i, smiles in enumerate(dataset.targets):
            fp = compute_fingerprint(smiles)
            sim = tanimoto_similarity(query_fp, fp)
            similarities.append((i, sim))

        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        top_k = similarities[:args.k]
        print(f"Top {args.k} entries similar to query {args.query}:")
        for idx, sim in top_k:
            print(f"Index: {idx}, SMILES: {dataset.targets[idx]}, Tanimoto: {sim:.4f}")

        print("\nRunning inference on selected entries:")
        for idx, sim in top_k:
            target_tokens, (ir_data, _), nmr_tokens, _ = dataset[idx]
            if ir_data is not None:
                ir_data = ir_data.to(device)
            nmr_tokens = nmr_tokens.to(device)
            prediction = greedy_decode(model, nmr_tokens, ir_data, tokenizer, device=device)
            print(f"Index: {idx}")
            print(f"Dataset SMILES: {dataset.targets[idx]}")
            print(f"Tanimoto similarity: {sim:.4f}")
            print(f"Predicted SMILES: {prediction[0]}")
            print("-"*40)


if __name__ == '__main__':
    main() 