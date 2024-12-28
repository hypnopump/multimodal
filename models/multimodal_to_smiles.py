import torch
import torch.nn as nn

from .spectral_encoder import MultimodalSpectralEncoder
from .transformer_decoder import SMILESDecoder

class MultiModalToSMILESModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_length=512,
        embed_dim=768,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        resample_size=1000,
        use_concat=True,
        verbose=True
    ):
        super().__init__()
        
        self.use_concat = use_concat
        self.verbose = verbose
        
        # Spectral encoder with verbose
        self.encoder = MultimodalSpectralEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            resample_size=resample_size,
            use_concat=use_concat,
            verbose=verbose
        )
        
        # Calculate decoder input dimension
        decoder_dim = embed_dim * 3 if use_concat else embed_dim
        
        # SMILES decoder with verbose
        self.decoder = SMILESDecoder(
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            memory_dim=decoder_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            verbose=verbose
        )

    def forward(self, nmr_data, ir_data, hsqc_data, target_seq=None, target_mask=None):
        if self.verbose:
            print("\n=== Starting Forward Pass ===")
            
        # Encode spectral inputs
        memory = self.encoder(nmr_data, ir_data, hsqc_data)
        
        if self.verbose:
            print("\n=== Starting Decoding ===")
            
        # Decode to SMILES
        logits = self.decoder(target_seq, memory, target_mask)
        
        if self.verbose:
            print("\n=== Forward Pass Complete ===")
            
        return logits 