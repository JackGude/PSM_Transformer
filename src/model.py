"""Transformer model definition for phoneme translation + PSV/PSM prediction.

 This module contains:
 - `PositionalEncoding`: standard sinusoidal positional encoding.
 - `PhonemeTransformer`: a two-headed `torch.nn.Transformer`.
   - Head 1 (`fc_out`): token-level logits for phoneme translation.
   - Head 2 (`aux_head`): sequence-level logits for a dense multi-label
     phoneme-shift vector (called PSV in the paper; often named PSM/psm in code).
     The auxiliary head pools encoder states (masked mean) into a single vector.
 - `create_masks`: builds padding masks and the decoder causal mask.

 Training/evaluation scripts import these definitions to ensure consistent model
 structure when saving/loading checkpoints.
 """

import torch
import torch.nn as nn
import math

# --- 1. Helper: Positional Encoding ---
# This is a standard, necessary part of any Transformer.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Input x: (seq_len, batch_size, embed_size)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# --- 2. The Two-Headed Transformer Model ---

class PhonemeTransformer(nn.Module):
    def __init__(self, 
                 num_phonemes,    # Size of your phoneme vocabulary
                 psv_size,        # Size of your PSM vector
                 d_model,         # Embedding size
                 nhead,           # Number of attention heads
                 num_encoders,    # Number of encoder layers
                 num_decoders,    # Number of decoder layers
                 d_ffn,           # Internal dimension of FFN
                 dropout=0.1):
        
        super(PhonemeTransformer, self).__init__()
        self.d_model = d_model
        
        # --- Core Components ---
        self.embedding = nn.Embedding(num_phonemes, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # We use batch_first=False (the default) as it's more stable
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoders,
            num_decoder_layers=num_decoders,
            dim_feedforward=d_ffn,
            dropout=dropout,
            batch_first=False
        )
        
        # --- Head 1: Translation Decoder ---
        self.fc_out = nn.Linear(d_model, num_phonemes)
        
        # --- Head 2: PSM Auxiliary Head ---
        self.aux_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, psv_size) 
        )

    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, tgt_mask):
        """
        src: (src_len, batch_size) - The Latin phonemes
        tgt: (tgt_len, batch_size) - The French/Spanish/etc. phonemes
        src_padding_mask: (batch_size, src_len)
        tgt_padding_mask: (batch_size, tgt_len)
        tgt_mask: (tgt_len, tgt_len)
        """
        
        # 1. Embed and encode source (Latin)
        # src_embed shape: (src_len, batch_size, d_model)
        src_embed = self.embedding(src) * math.sqrt(self.d_model)
        src_embed_pos = self.pos_encoder(src_embed) 
        
        # 2. Embed and encode target (French, etc.)
        # tgt_embed shape: (tgt_len, batch_size, d_model)
        tgt_embed = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_embed_pos = self.pos_encoder(tgt_embed)

        # 3. Get Encoder Output (The "Latent Space")
        # memory shape: (src_len, batch_size, d_model)
        memory = self.transformer.encoder(
            src_embed_pos, 
            src_key_padding_mask=src_padding_mask
        )
        
        # 4. Get Decoder Output
        # transformer_out shape: (tgt_len, batch_size, d_model)
        transformer_out = self.transformer.decoder(
            tgt_embed_pos,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        # --- 5. Get Outputs from Both Heads ---
        
        # Head 1: Translation Output
        # We must transpose from (tgt_len, batch_size, d_model) 
        # to (batch_size, tgt_len, d_model) for the loss function.
        translation_logits = self.fc_out(transformer_out.transpose(0, 1)) 
        
        # Head 2: PSM Output
        # Global average pooling over valid encoder time steps
        if src_padding_mask is not None:
            # src_padding_mask: (batch_size, src_len) with True where padded
            valid_mask = (~src_padding_mask).transpose(0, 1).unsqueeze(-1)  # (src_len, batch_size, 1)
            masked_memory = memory * valid_mask
            counts = valid_mask.sum(dim=0).clamp(min=1.0)
            pooled = masked_memory.sum(dim=0) / counts
        else:
            pooled = memory.mean(dim=0)
        psm_logits = self.aux_head(pooled)

        return translation_logits, psm_logits

# --- 3. Helper: Mask Creation Function ---

def create_masks(src, tgt, pad_idx, device="cpu"):
    """
    Creates all necessary masks for the Transformer.
    src: (src_len, batch_size)
    tgt: (tgt_len, batch_size)
    """
    # The mask must be (batch_size, seq_len)
    src_padding_mask = (src == pad_idx).transpose(0, 1).to(device)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1).to(device)

    # (tgt_len, tgt_len)
    # This is the "causal" mask to prevent the decoder
    # from cheating and looking at future tokens.
    tgt_len = tgt.shape[0] # Get seq_len from (seq_len, batch_size)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, dtype=torch.bool).to(device)

    return src_padding_mask, tgt_padding_mask, tgt_mask