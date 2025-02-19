# =======================
# File: train_autoregressive.py
# =======================
"""
Script to train a MultiModalToSMILESModel using either:
- Preprocessed .pt files created by create_training_data.py, or
- Parquet files (loaded in one go, no row-group chunking).

Key Steps:
1) Loads spectral + SMILES data
2) Tokenizes SMILES
3) Demonstrates a basic training loop with teacher forcing
4) Shows a minimal inference (greedy decode) function
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
from datetime import datetime
from sklearn.model_selection import train_test_split
import json
from rdkit import Chem
from rdkit.Chem import rdFMCS
from pathlib import Path
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
from tqdm import tqdm
import pandas as pd
import pyarrow.dataset as ds
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse
import time
import pyarrow.parquet as pq
from pprint import pprint

# Import our custom tokenizer
from models.smiles_tokenizer import SmilesTokenizer
from models.multimodal_to_smiles import MultiModalToSMILESModel

current_dir = os.path.dirname(os.path.realpath(__file__))
vocab_path = os.path.join(current_dir, 'vocab.txt')
tokenizer = SmilesTokenizer(vocab_file=vocab_path)

# -------------------------------------------------------------------------
# Warmup + Cosine Annealing With Restarts Scheduler
# -------------------------------------------------------------------------
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Combines linear warmup with cosine annealing and warm restarts.
    """
    def __init__(self, optimizer, warmup_steps, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.T_0_initial = T_0
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.completed_warmup = False
        self.n_restarts = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        
        # Cosine annealing with warm restarts
        if not self.completed_warmup:
            self.completed_warmup = True
            self.T_cur = 0
        
        # Check for restart
        if self.T_cur >= self.T_0:
            self.T_cur = 0
            self.T_0 = self.T_0 * self.T_mult
            self.n_restarts += 1
        
        progress = self.T_cur / self.T_0
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        self.T_cur += 1
        
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_decay
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self._last_lr = self.get_lr()
        
        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group['lr'] = lr


# -------------------------------------------------------------------------
# Dataset / DataLoader for Memory-Mapped Binary Files
# -------------------------------------------------------------------------
class SpectralSmilesDataset(Dataset):
    """
    A PyTorch Dataset that reads from memory-mapped binary files:
      - spectra_data.bin  (float32 arrays for IR/H-NMR/C-NMR + domains)
      - smiles_data.bin   (raw UTF-8 bytes for SMILES)
      - spectra_index.npy (offsets/lengths for each row)
    """
    def __init__(self, data_dir, tokenizer, max_len=128):
        """
        data_dir: directory containing:
          - spectra_data.bin
          - smiles_data.bin
          - spectra_index.npy
        tokenizer: SmilesTokenizer instance
        max_len: maximum SMILES token length
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Load index, which is shape [num_rows, 14] for IR/H-NMR/C-NMR + domains + SMILES
        index_path = self.data_dir / "spectra_index.npy"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found at {index_path}")
        self.index = np.load(index_path)  # shape: (num_rows, 14)

        # Memory-map the big float32 spectra file
        spectra_bin_path = self.data_dir / "spectra_data.bin"
        if not spectra_bin_path.exists():
            raise FileNotFoundError(f"Spectra data file not found at {spectra_bin_path}")
        self.spectra_mmap = np.memmap(spectra_bin_path, dtype=np.float32, mode='r')

        # Memory-map the SMILES file (raw bytes)
        smiles_bin_path = self.data_dir / "smiles_data.bin"
        if not smiles_bin_path.exists():
            raise FileNotFoundError(f"SMILES data file not found at {smiles_bin_path}")
        self.smiles_mmap = np.memmap(smiles_bin_path, dtype=np.uint8, mode='r')

        print(f"[Dataset] MemoryMappedSpectralDataset initialized:")
        print(f"          Index shape = {self.index.shape}")
        print(f"          Found {len(self.index)} samples total.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
        Each row in self.index has 14 columns:
          0: IR_data_off,    1: IR_data_len,
          2: IR_dom_off,     3: IR_dom_len,
          4: HNMR_data_off,  5: HNMR_data_len,
          6: HNMR_dom_off,   7: HNMR_dom_len,
          8: CNMR_data_off,  9: CNMR_data_len,
         10: CNMR_dom_off,  11: CNMR_dom_len,
         12: SMILES_off,    13: SMILES_len
        """
        (
            ir_data_off, ir_data_len,
            ir_dom_off,  ir_dom_len,
            hnm_data_off, hnm_data_len,
            hnm_dom_off,  hnm_dom_len,
            cnm_data_off, cnm_data_len,
            cnm_dom_off,  cnm_dom_len,
            smiles_off,   smiles_len
        ) = self.index[idx]

        # -------------------------
        # Retrieve IR data + domain
        # -------------------------
        if ir_data_off == -1 or ir_data_len == 0:
            ir_tuple = None
        else:
            # Float32 slice from the memory-mapped array
            ir_data = self.spectra_mmap[ir_data_off : ir_data_off + ir_data_len]
            # Domain
            if ir_dom_off == -1 or ir_dom_len == 0:
                ir_dom = None
            else:
                ir_dom = self.spectra_mmap[ir_dom_off : ir_dom_off + ir_dom_len]
            # Convert to torch tensors
            ir_data_t = torch.from_numpy(ir_data.copy()) if ir_data_len > 0 else None
            ir_dom_t  = torch.from_numpy(ir_dom.copy())  if ir_dom is not None else None
            ir_tuple  = (ir_data_t, ir_dom_t)

        # -------------------------
        # Retrieve H-NMR data + domain
        # -------------------------
        if hnm_data_off == -1 or hnm_data_len == 0:
            h_nmr_tuple = None
        else:
            h_nmr_data = self.spectra_mmap[hnm_data_off : hnm_data_off + hnm_data_len]
            if hnm_dom_off == -1 or hnm_dom_len == 0:
                h_nmr_dom = None
            else:
                h_nmr_dom = self.spectra_mmap[hnm_dom_off : hnm_dom_off + hnm_dom_len]
            h_nmr_tuple = (
                torch.from_numpy(h_nmr_data.copy()),
                torch.from_numpy(h_nmr_dom.copy()) if h_nmr_dom is not None else None
            )

        # -------------------------
        # Retrieve C-NMR data + domain
        # -------------------------
        if cnm_data_off == -1 or cnm_data_len == 0:
            c_nmr_tuple = None
        else:
            c_nmr_data = self.spectra_mmap[cnm_data_off : cnm_data_off + cnm_data_len]
            if cnm_dom_off == -1 or cnm_dom_len == 0:
                c_nmr_dom = None
            else:
                c_nmr_dom = self.spectra_mmap[cnm_dom_off : cnm_dom_off + cnm_dom_len]
            c_nmr_tuple = (
                torch.from_numpy(c_nmr_data.copy()),
                torch.from_numpy(c_nmr_dom.copy()) if c_nmr_dom is not None else None
            )

        # -------------------------
        # Retrieve SMILES string
        # -------------------------
        if smiles_off == -1 or smiles_len == 0:
            smiles_str = ""
        else:
            smiles_bytes = self.smiles_mmap[smiles_off : smiles_off + smiles_len]
            smiles_str = smiles_bytes.tobytes().decode('utf-8')

        # -------------------------
        # Tokenize SMILES
        # -------------------------
        tokens = self.tokenizer.encode(
            smiles_str,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True
        )
        tokens = torch.tensor(tokens, dtype=torch.long)

        return tokens, ir_tuple, h_nmr_tuple, c_nmr_tuple


# -------------------------------------------------------------------------
# Dataset / DataLoader for Parquet Files (No Chunking)
# -------------------------------------------------------------------------
class ParquetSpectralDataset(Dataset):
    """
    A PyTorch Dataset that reads Parquet files using pandas.
    """
    def __init__(self, data_dir, tokenizer, max_len=128):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Load metadata
        meta_path = self.data_dir / "meta_data/meta_data_dict.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")
        with open(meta_path) as f:
            self.meta_data = json.load(f)

        # Store domain information
        self.ir_domain = torch.tensor(self.meta_data["ir_spectra"]["dimensions"], dtype=torch.float32)
        self.h_nmr_domain = torch.tensor(self.meta_data["h_nmr_spectra"]["dimensions"], dtype=torch.float32)
        self.c_nmr_domain = torch.tensor(self.meta_data["c_nmr_spectra"]["dimensions"], dtype=torch.float32)

        # Find and load all parquet files
        print("[Dataset] Looking for parquet files...")
        self.parquet_files = sorted(self.data_dir.glob("*.parquet"))
        if not self.parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")
        print(f"[Dataset] Found {len(self.parquet_files)} parquet files")

        # Load all data into memory
        print("[Dataset] Loading parquet files...")
        dfs = []
        for file in tqdm(self.parquet_files, desc="Loading parquet files"):
            df = pd.read_parquet(
                file,
                columns=['smiles', 'ir_spectra', 'h_nmr_spectra', 'c_nmr_spectra'],
                engine='pyarrow'
            )
            dfs.append(df)
        
        self.data = pd.concat(dfs, ignore_index=True)
        print(f"[Dataset] Loaded {len(self.data)} total rows")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Tokenize SMILES
        smiles = row['smiles']
        tokens = self.tokenizer.encode(
            smiles,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True
        )

        def to_tensor(x, spectrum_type):
            if x is None:
                return None
            try:
                tensor = torch.tensor(x, dtype=torch.float32)
                if tensor.dim() == 1:
                    if spectrum_type == 'ir':
                        return (tensor, self.ir_domain)
                    elif spectrum_type == 'h_nmr':
                        return (tensor, self.h_nmr_domain)
                    elif spectrum_type == 'c_nmr':
                        return (tensor, self.c_nmr_domain)
                return tensor
            except Exception as e:
                print(f"Warning: Error converting {spectrum_type} data to tensor: {e}")
                return None

        # Convert spectra to tuples of (data, domain)
        ir_spectra = to_tensor(row['ir_spectra'], 'ir')
        h_nmr_spectra = to_tensor(row['h_nmr_spectra'], 'h_nmr')
        c_nmr_spectra = to_tensor(row['c_nmr_spectra'], 'c_nmr')

        return tokens, ir_spectra, h_nmr_spectra, c_nmr_spectra


# -------------------------------------------------------------------------
# Collate Function - Moved outside to be picklable
# -------------------------------------------------------------------------
def collate_fn(batch):
    """
    Custom collate: pad tokens, preserve spectral data tuples.
    """
    # Unzip the batch into separate lists
    all_tokens, all_ir, all_h_nmr, all_c_nmr = zip(*batch)
    
    # Helper function to stack spectral data tuples
    def maybe_stack_with_domain(items):
        if items[0] is not None:
            # Stack data tensors along batch dimension
            data = torch.stack([item[0] for item in items], dim=0)
            # Use first domain tensor (they're all the same)
            domain = items[0][1]
            return (data, domain)
        return None

    # Stack spectral data preserving tuple structure
    ir_batch = maybe_stack_with_domain(all_ir) if all_ir[0] is not None else None
    h_nmr_batch = maybe_stack_with_domain(all_h_nmr) if all_h_nmr[0] is not None else None
    c_nmr_batch = maybe_stack_with_domain(all_c_nmr) if all_c_nmr[0] is not None else None

    # Pad tokens
    max_len = max(len(t) for t in all_tokens)
    padded_tokens = []
    for seq in all_tokens:
        pad_amount = max_len - len(seq)
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        if pad_amount > 0:
            pad_tensor = torch.full((pad_amount,), tokenizer.pad_token_id, dtype=torch.long)
            seq_tensor = torch.cat([seq_tensor, pad_tensor], dim=0)
        padded_tokens.append(seq_tensor)
    token_batch = torch.stack(padded_tokens, dim=0)

    return token_batch, ir_batch, h_nmr_batch, c_nmr_batch


def create_data_loaders(tokenizer, config):
    print("\n[DataLoader] Creating data loaders...")
    if config['data']['use_parquet']:
        # If using Parquet, keep the old approach
        dataset = ParquetSpectralDataset(
            data_dir=config['data']['data_dir'],
            tokenizer=tokenizer,
            max_len=config['model']['max_seq_length']
        )
    else:
        # Use the new memory-mapped dataset
        dataset = SpectralSmilesDataset(
            data_dir=config['data']['binary_dir'],
            tokenizer=tokenizer,
            max_len=config['model']['max_seq_length']
        )
    
    print(f"[DataLoader] Total dataset size: {len(dataset)}")
    
    # Split indices
    print("[DataLoader] Splitting dataset into train/val/test...")
    all_indices = list(range(len(dataset)))
    train_val_indices, test_indices = train_test_split(
        all_indices, 
        test_size=config['data'].get('test_size', 20), 
        random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_val_indices, 
        test_size=config['data'].get('val_size', 0.1), 
        random_state=42
    )
    
    print(f"[DataLoader] Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Create loaders with simple collate function
    print("[DataLoader] Creating train loader...")
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print("[DataLoader] Creating validation loader...")
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print("[DataLoader] Creating test loader...")
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=config['training'].get('test_batch_size', 1),
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print("[DataLoader] Data loaders created successfully")
    return train_loader, val_loader, test_loader


# -------------------------------------------------------------------------
# Setup: Config, Tokenizer, Model, Data Loaders
# -------------------------------------------------------------------------
def load_config(config_path=None):
    """Load config from yaml file, falling back to defaults if not specified"""
    default_config = {
        'model': {
            'max_seq_length': 128,
            'embed_dim': 768,
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1,
            'resample_size': 1000,
            'use_concat': True
        },
        'training': {
            'batch_size': 32,
            'test_batch_size': 1,
            'num_epochs': 1,
            'learning_rate': 1.0e-4,
            'min_learning_rate': 1.0e-6,
            'validation_frequency': 500,
            'logging_frequency': 100,
            'save_frequency': 1000,
            'generate_during_training': False,
            'save_local': False  # Default to not saving locally
        },
        'scheduler': {
            'warmup_steps': 100,
            'T0': 5,
            'T_mult': 2
        },
        'data': {
            'use_parquet': False,
            'data_dir': "data_extraction/multimodal_spectroscopic_dataset",
            'binary_dir': "training_binaries",
            'preprocessed': False,
            'test_size': 20,
            'val_size': 0.1
        },
        'wandb': {
            'project': "smiles-generation",
            'base_run_name': "smiles_gen",
            'log_examples': True
        }
    }
    
    if config_path:
        with open(config_path, 'r') as f:
            custom_config = yaml.safe_load(f)
            # Recursively update default config with custom values
            def update_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = update_dict(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d
            update_dict(default_config, custom_config)
    
    return default_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train SMILES generation model')
    parser.add_argument('--config', type=str, help='Path to config file')
    return parser.parse_args()

def get_domain_ranges(meta_data):
    """Extract domain ranges from metadata"""
    ir_range = [
        min(meta_data["ir_spectra"]["dimensions"]),
        max(meta_data["ir_spectra"]["dimensions"])
    ]
    h_nmr_range = [
        min(meta_data["h_nmr_spectra"]["dimensions"]),
        max(meta_data["h_nmr_spectra"]["dimensions"])
    ]
    c_nmr_range = [
        min(meta_data["c_nmr_spectra"]["dimensions"]),
        max(meta_data["c_nmr_spectra"]["dimensions"])
    ]
    hsqc_h_range = [
        min(meta_data["hsqc_nmr_spectrum"]["dimensions"]["h"]),
        max(meta_data["hsqc_nmr_spectrum"]["dimensions"]["h"])
    ]
    hsqc_c_range = [
        min(meta_data["hsqc_nmr_spectrum"]["dimensions"]["c"]),
        max(meta_data["hsqc_nmr_spectrum"]["dimensions"]["c"])
    ]
    return ir_range, h_nmr_range, c_nmr_range, hsqc_h_range, hsqc_c_range

# Add this main function to contain the training code
def main():
    print("\n[Main] Starting training script...")
    args = parse_args()
    
    print("[Main] Loading configuration...")
    config = load_config(args.config)
    print(f"[Main] Loaded config with {len(config)} sections")

    print("\n[Main] Setting up model parameters...")
    max_seq_length = config['model']['max_seq_length']
    batch_size = config['training']['batch_size']
    embed_dim = config['model']['embed_dim']
    num_heads = config['model']['num_heads']
    num_layers = config['model']['num_layers']
    dropout = config['model']['dropout']
    resample_size = config['model']['resample_size']

    PAD_TOKEN_ID = tokenizer.pad_token_id
    BOS_TOKEN_ID = tokenizer.cls_token_id
    EOS_TOKEN_ID = tokenizer.sep_token_id

    print("\n[Main] Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"[Main] Found {torch.cuda.device_count()} CUDA devices:")
        for i in range(torch.cuda.device_count()):
            print(f"  - {torch.cuda.get_device_name(i)}")
    else:
        print("[Main] No CUDA devices found, using CPU")
    print(f"[Main] Using device: {device}")

    print("\n[Main] Initializing model...")
    
    # Get domain ranges from dataset's metadata
    if config['data']['use_parquet']:
        meta_path = Path(config['data']['data_dir']) / "meta_data/meta_data_dict.json"
        with open(meta_path) as f:
            meta_data = json.load(f)
        domain_ranges = get_domain_ranges(meta_data)
    else:
        domain_ranges = None  # Use defaults for binary dataset
    
    model = MultiModalToSMILESModel(
        vocab_size=len(tokenizer),
        max_seq_length=max_seq_length,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        resample_size=resample_size,
        domain_ranges=domain_ranges,
        verbose=False,
        use_concat=config['model']['use_concat']
    ).to(device)
    print("[Main] Model initialized successfully")



    print("\n[Main] Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        tokenizer=tokenizer,
        config=config
    )

    print("\n[Main] Initializing wandb...")
    run_name = (
        f"{config['wandb']['base_run_name']}_"
        f"enc_d{embed_dim}_"
        f"enc_h{num_heads}_"
        f"dec_l{num_layers}_"
        f"bs{batch_size}_"
        f"lr{config['training']['learning_rate']}_warm{config['scheduler']['warmup_steps']}_"
        f"{datetime.now().strftime('%m%d_%H%M')}"
    )

    wandb.init(
        project=config['wandb']['project'],
        name=run_name,
        config=config
    )
    print("[Main] wandb initialized successfully")

    # Add this section to log model size
    print("\n[Main] Calculating model size...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # Size in MB

    print(f"[Main] Total parameters: {total_params:,}")
    print(f"[Main] Trainable parameters: {trainable_params:,}")
    print(f"[Main] Model size: {param_size:.2f} MB")

    # Log to wandb
    wandb.run.summary.update({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": param_size
    })

    print("\n[Main] Setting up training components...")
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = WarmupCosineLR(
        optimizer,
        warmup_steps=config['scheduler']['warmup_steps'],
        T_0=config['scheduler']['T0'] * len(train_loader),
        T_mult=config['scheduler']['T_mult'],
        eta_min=config['training']['min_learning_rate']
    )

    print("\n[Main] Creating checkpoint directory...")
    save_dir = Path('checkpoints') / datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Main] Checkpoint directory created at {save_dir}")

    print("\n[Main] Starting training loop...")
    NUM_EPOCHS = config['training']['num_epochs']
    validation_frequency = config['training']['validation_frequency']
    verbose = False
    
    # Add timing stats
    batch_times = []
    data_loading_times = []
    forward_times = []
    backward_times = []

    best_val_loss = float('inf')
    epoch_loss = 0
    num_batches = 0
    global_step = 0

    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------
    def evaluate_predictions(predictions, targets, verbose=False):
        """
        Evaluate model predictions vs. targets using canonical SMILES.
        Returns detailed metrics and optionally prints examples.
        """
        from rdkit import DataStructs, Chem
        from rdkit.Chem import AllChem
        
        detailed_results = []
        
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            result = {
                'prediction': pred,
                'target': target,
                'valid': False,
                'valid_target': False,
                'exact_match': False,
                'tanimoto': 0.0,
                '#mcs/#target': 0.0,
                'ecfp6_iou': 0.0
            }
            
            # Remove spaces before creating molecules
            pred_no_spaces = pred.replace(" ", "")
            target_no_spaces = target.replace(" ", "")
            
            # Convert to RDKit molecules
            mol_pred = Chem.MolFromSmiles(pred_no_spaces)
            mol_target = Chem.MolFromSmiles(target_no_spaces)

            result['valid'] = mol_pred is not None
            result['valid_target'] = mol_target is not None

            if result['valid'] and result['valid_target']:
                # Get canonical SMILES
                canon_pred = Chem.MolToSmiles(mol_pred, canonical=True)
                canon_target = Chem.MolToSmiles(mol_target, canonical=True)
                result['exact_match'] = canon_pred == canon_target

                # Standard Tanimoto similarity
                fp_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, 2)
                fp_target = AllChem.GetMorganFingerprintAsBitVect(mol_target, 2)
                tanimoto = DataStructs.TanimotoSimilarity(fp_pred, fp_target)
                result['tanimoto'] = tanimoto

                # ECFP6 IoU: ~ substructures predicted / substructures present :: ECFP6 has radius=3
                fp3_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, radius=3, nBits=1024)
                fp3_traget = AllChem.GetMorganFingerprintAsBitVect(mol_target, radius=3, nBits=1024)
                intersection = sum((fp3_pred & fp3_traget))  # Count of common bits
                union = sum((fp3_pred | fp3_traget))  # Count of total unique bits
                result['ecfp6_iou'] = intersection / union

                # MCS: largest common substructure: calc num_atoms(mcs) / num_atoms(target)
                mcs_result = Chem.rdFMCS.FindMCS([mol_pred, mol_target])
                mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
                mcs_num_atoms = mcs_mol.GetNumAtoms()
                target_num_atoms = mol_target.GetNumAtoms()
                result['#mcs/#target'] = mcs_num_atoms / target_num_atoms

                if verbose and i < 5:  # Print first 5 examples
                    print(f"\nCanonical SMILES comparison:")
                    print(f"Target (canonical):     {canon_target}")
                    print(f"Prediction (canonical): {canon_pred}")
            
            detailed_results.append(result)
            
            # Print some examples if verbose
            if verbose and i < 5:
                print(f"\nExample {i+1}:")
                pprint(result)
        
        # aggregate batch-wise metrics
        metrics = {
            'valid_smiles': np.mean([r['valid'] for r in detailed_results]),
            'exact_match': np.mean([r['exact_match'] for r in detailed_results]),
        }
        chem_results = list(filter(lambda x: x['valid'] and x['valid_target'], detailed_results))
        metrics['avg_tanimoto'] = 0.0
        metrics['avg_exact_match'] = 0.0
        metrics['avg_ecfp6_iou'] = 0.0
        metrics['avg_#mcs/#target'] = 0.0
        if chem_results:
            metrics['avg_tanimoto'] = np.mean([r['tanimoto'] for r in chem_results])
            metrics['avg_exact_match'] = np.mean([r['exact_match'] for r in chem_results])
            metrics['avg_ecfp6_iou'] = np.mean([r['ecfp6_iou'] for r in chem_results])
            metrics['avg_#mcs/#target'] = np.mean([r['exact_#mcs/#target'] for r in chem_results])
        
        return metrics, detailed_results


    def greedy_decode(model, nmr_data, ir_data, c_nmr_data, max_len=128):
        """
        Simple greedy decoding for SMILES generation.
        """
        model.eval()
        with torch.no_grad():
            # Get batch size from input data
            batch_size = 1  # Default
            if nmr_data is not None:
                batch_size = nmr_data[0].size(0) if isinstance(nmr_data, tuple) else nmr_data.size(0)
            elif ir_data is not None:
                batch_size = ir_data[0].size(0) if isinstance(ir_data, tuple) else ir_data.size(0)
            elif c_nmr_data is not None:
                batch_size = c_nmr_data[0].size(0) if isinstance(c_nmr_data, tuple) else c_nmr_data.size(0)

            # Start tokens for each sequence in the batch
            current_token = torch.tensor([[BOS_TOKEN_ID]] * batch_size, device=device)
            
            # Handle spectral data tuples properly
            if nmr_data is not None:
                if isinstance(nmr_data, tuple):
                    # If it's already a tuple of (data, domain), keep as is
                    pass
                else:
                    # Add batch dimension if needed
                    if nmr_data.dim() == 1:
                        nmr_data = nmr_data.unsqueeze(0)
                    # Create domain tensor if needed (should be provided by dataset)
                    raise ValueError("NMR data must be provided as (data, domain) tuple")
                    
            if ir_data is not None:
                if isinstance(ir_data, tuple):
                    pass
                else:
                    if ir_data.dim() == 1:
                        ir_data = ir_data.unsqueeze(0)
                    raise ValueError("IR data must be provided as (data, domain) tuple")
                    
            if c_nmr_data is not None:
                if isinstance(c_nmr_data, tuple):
                    pass
                else:
                    if c_nmr_data.dim() == 1:
                        c_nmr_data = c_nmr_data.unsqueeze(0)
                    raise ValueError("C-NMR data must be provided as (data, domain) tuple")
            
            # Encode
            memory = model.encoder(nmr_data, ir_data, c_nmr_data)
            
            # Initialize storage for generated tokens
            generated_sequences = [[] for _ in range(batch_size)]
            for seq in generated_sequences:
                seq.append(BOS_TOKEN_ID)
            
            # Use the decoder's max_seq_length as the limit
            max_len = min(max_len, model.decoder.max_seq_length)
            
            finished_sequences = [False] * batch_size
            
            for _ in range(max_len):
                logits = model.decoder(current_token, memory)
                next_token = logits[:, -1:].argmax(dim=-1)
                
                # Update each sequence
                for i in range(batch_size):
                    if not finished_sequences[i]:
                        token = next_token[i].item()
                        generated_sequences[i].append(token)
                        if token == EOS_TOKEN_ID:
                            finished_sequences[i] = True
                
                # Stop if all sequences are finished
                if all(finished_sequences):
                    break
                
                current_token = torch.cat([current_token, next_token], dim=1)
            
            # Convert to tensor
            max_seq_len = max(len(seq) for seq in generated_sequences)
            padded_sequences = []
            for seq in generated_sequences:
                # Pad sequence to max length
                padded_seq = seq + [PAD_TOKEN_ID] * (max_seq_len - len(seq))
                padded_sequences.append(padded_seq)
            
            return torch.tensor(padded_sequences, device=device)


    def validate(model, val_loader, criterion, tokenizer):
        """Validation using teacher forcing and comparing exact matches"""
        model.eval()
        total_loss = 0
        num_batches = 0
        exact_matches = 0
        total_sequences = 0
        matching_pairs = []  # Add this to track matching SMILES pairs
        
        with torch.no_grad():
            for tgt_tokens, ir, h_nmr, c_nmr in val_loader:
                tgt_tokens = tgt_tokens.to(device)
                
                # Handle spectral data tuples
                if ir is not None:
                    if isinstance(ir, tuple):
                        ir = (ir[0].to(device), ir[1].to(device))
                    else:
                        ir = ir.to(device)
                        
                if h_nmr is not None:
                    if isinstance(h_nmr, tuple):
                        h_nmr = (h_nmr[0].to(device), h_nmr[1].to(device))
                    else:
                        h_nmr = h_nmr.to(device)
                        
                if c_nmr is not None:
                    if isinstance(c_nmr, tuple):
                        c_nmr = (c_nmr[0].to(device), c_nmr[1].to(device))
                    else:
                        c_nmr = c_nmr.to(device)
                
                # Regular forward pass with teacher forcing
                T = tgt_tokens.shape[1]
                mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_tokens.device), 1)
                
                logits = model(h_nmr, ir, c_nmr, target_seq=tgt_tokens[:, :-1], target_mask=mask[:-1, :-1])
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_tokens[:, 1:].reshape(-1))
                
                # Get predictions from logits and compare with targets
                pred_tokens = logits.argmax(dim=-1)  # Shape: [batch_size, seq_len]
                
                # Compare predictions from teacher forcing
                for pred, target in zip(pred_tokens, tgt_tokens[:, 1:]):  # Skip BOS token in target
                    pred_smiles = tokenizer.decode(pred.tolist(), skip_special_tokens=True)
                    target_smiles = tokenizer.decode(target.tolist(), skip_special_tokens=True)
                    
                    # Remove spaces and compare canonical SMILES
                    pred_no_spaces = pred_smiles.replace(" ", "")
                    target_no_spaces = target_smiles.replace(" ", "")
                    
                    mol_pred = Chem.MolFromSmiles(pred_no_spaces)
                    mol_target = Chem.MolFromSmiles(target_no_spaces)
                    
                    if mol_pred is not None and mol_target is not None:
                        canon_pred = Chem.MolToSmiles(mol_pred, canonical=True)
                        canon_target = Chem.MolToSmiles(mol_target, canonical=True)
                        if canon_pred == canon_target:
                            exact_matches += 1
                            matching_pairs.append({
                                'predicted': canon_pred,
                                'target': canon_target
                            })
                
                total_loss += loss.item()
                num_batches += 1
                total_sequences += tgt_tokens.size(0)
        
        return {
            'val_loss': total_loss / num_batches,
            'val_exact_match': exact_matches / total_sequences,
            'matching_pairs': matching_pairs[:10]  # Store only first 10 matches to avoid excessive logging
        }


    def log_validation_results(val_metrics, global_step):
        """Log validation metrics and matching SMILES pairs to wandb"""
        # Basic metrics
        log_dict = {
            "val_loss": val_metrics['val_loss'],
            "val_exact_match": val_metrics['val_exact_match'],
        }
        
        # Add matching pairs as individual strings
        for i, pair in enumerate(val_metrics['matching_pairs']):
            log_dict[f"val_match_{i}_pred"] = pair['predicted']
            log_dict[f"val_match_{i}_target"] = pair['target']
        
        wandb.log(log_dict, step=global_step)


    def evaluate_on_test(model, test_loader, tokenizer, device):
        """Evaluate on test set using greedy decoding"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for tgt_tokens, ir, h_nmr, c_nmr in tqdm(test_loader, desc="Testing"):
                tgt_tokens = tgt_tokens.to(device)
                
                # Handle spectral data tuples
                if ir is not None:
                    if isinstance(ir, tuple):
                        ir = (ir[0].to(device), ir[1].to(device))
                    else:
                        ir = ir.to(device)
                
                if h_nmr is not None:
                    if isinstance(h_nmr, tuple):
                        h_nmr = (h_nmr[0].to(device), h_nmr[1].to(device))
                    else:
                        h_nmr = h_nmr.to(device)
                
                if c_nmr is not None:
                    if isinstance(c_nmr, tuple):
                        c_nmr = (c_nmr[0].to(device), c_nmr[1].to(device))
                    else:
                        c_nmr = c_nmr.to(device)
                
                # Use greedy decode with proper tensor handling
                pred_tokens = greedy_decode(model, h_nmr, ir, c_nmr)
                
                # Decode predictions and targets
                for i in range(pred_tokens.size(0)):
                    pred_smiles = tokenizer.decode(pred_tokens[i].tolist(), skip_special_tokens=True)
                    target_smiles = tokenizer.decode(tgt_tokens[i].tolist(), skip_special_tokens=True)
                    
                    all_predictions.append(pred_smiles)
                    all_targets.append(target_smiles)
        
        return all_predictions, all_targets


    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    model.train()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        epoch_loss = 0
        num_batches = 0
        
        # Calculate total batches for percentage tracking
        total_batches = len(train_loader)
        log_interval = max(1, total_batches // 20)  # Log ~20 times per epoch
        
        for batch in train_loader:
            batch_start_time = time.time()
            
            # Unpack the batch data correctly
            tgt_tokens, ir, h_nmr, c_nmr = batch
            
            # Get the batch data
            tgt_tokens = tgt_tokens.to(device)

            # Handle spectral data tuples
            if ir is not None:
                if isinstance(ir, tuple):
                    ir = (ir[0].to(device), ir[1].to(device))
                else:
                    ir = ir.to(device)

            if h_nmr is not None:
                if isinstance(h_nmr, tuple):
                    h_nmr = (h_nmr[0].to(device), h_nmr[1].to(device))
                else:
                    h_nmr = h_nmr.to(device)

            if c_nmr is not None:
                if isinstance(c_nmr, tuple):
                    c_nmr = (c_nmr[0].to(device), c_nmr[1].to(device))
                else:
                    c_nmr = c_nmr.to(device)

            # Forward pass
            T = tgt_tokens.shape[1]
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_tokens.device), 1)
            logits = model(h_nmr, ir, c_nmr, target_seq=tgt_tokens[:, :-1], target_mask=mask[:-1, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_tokens[:, 1:].reshape(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            current_lr = scheduler.get_lr()[0]
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Simple progress logging
            if num_batches % log_interval == 0:
                avg_loss = epoch_loss / num_batches
                progress = (num_batches / total_batches) * 100
                print(f"Progress: {progress:3.0f}% | Batch {num_batches}/{total_batches} | "
                      f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

            # Log to wandb
            wandb.log({
                "batch_loss": loss.item(),
                "learning_rate": current_lr,
                "epoch": epoch + 1,
                "global_step": global_step,
            }, step=global_step)

            # Periodic validation
            if global_step % validation_frequency == 0:
                print("\nRunning validation...")
                val_metrics = validate(model, val_loader, criterion, tokenizer)
                log_validation_results(val_metrics, global_step)
                print(f"Validation - Loss: {val_metrics['val_loss']:.4f} | "
                      f"Exact Match: {val_metrics['val_exact_match']:.2%}")
                
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    print(f"New best validation loss: {best_val_loss:.4f}")
                    
                    checkpoint = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_metrics': val_metrics,
                    }
                    
                    # Save locally if configured
                    if config['training'].get('save_local', False):
                        torch.save(checkpoint, save_dir / 'best_model.pt')
                        print(f"Saved checkpoint locally to {save_dir / 'best_model.pt'}")
                    
                    # Always save to wandb
                    artifact = wandb.Artifact(
                        name="model", 
                        type="model",
                        description=f"Model checkpoint at step {global_step} with val_loss: {best_val_loss:.4f}"
                    )

                    wandb.log_artifact(artifact, aliases=["latest", f"step_{global_step}"])
                    print(f"Saved checkpoint to wandb (val_loss: {best_val_loss:.4f})")
                model.train()

        # End of epoch logging
        avg_epoch_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch+1} completed | Average Loss: {avg_epoch_loss:.4f}")

    # -------------------------------------------------------------------------
    # Final Evaluation on Test Set
    # -------------------------------------------------------------------------
    print("\nRunning final evaluation on test set...")
    all_predictions, all_targets = evaluate_on_test(model, test_loader, tokenizer, device)

    results = evaluate_predictions(all_predictions, all_targets)
    wandb.log({
        "test_exact_match": results['exact_match'],
        "test_valid_smiles": results['valid_smiles'],
        "test_avg_tanimoto": results['avg_tanimoto']
    })

    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nTest Results:")
    print(f"Exact Match: {results['exact_match']:.2%}")
    print(f"Valid SMILES: {results['valid_smiles']:.2%}")
    print(f"Avg Tanimoto: {results['avg_tanimoto']:.3f}")

    wandb.finish()

# Add this guard at the bottom of the file
if __name__ == '__main__':
    # Optional: Add freeze_support() if you plan to create executables
    # from multiprocessing import freeze_support
    # freeze_support()
    
    main()
