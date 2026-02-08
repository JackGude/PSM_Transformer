"""Visualize cross-attention for a single (Latin -> target language) word example.

 This script loads a trained checkpoint and instruments the last decoder layer's
 cross-attention (`multihead_attn`) to capture attention weights. It then:
 - constructs a single-example input from a provided Latin phoneme string and
   language token,
 - uses the dataset to fetch the corresponding ground-truth target phonemes when
   available (otherwise falls back to a dummy target),
 - runs a forward pass and plots a heatmap of cross-attention weights between
   output tokens (target phonemes) and input tokens (language token + Latin).

 Output: `results/visuals/attention_<LANG>_<WORD>.png`.
 """

# src/visualize_attention.py
import argparse
import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from src.model import PhonemeTransformer, create_masks
except ImportError:
    from model import PhonemeTransformer, create_masks


# --- 2. THE ATTENTION PROBE ---
class AttentionProbe(nn.Module):
    """
    Wraps a MultiheadAttention module to capture its weights during forward pass.
    """
    def __init__(self, original_module):
        super().__init__()
        self.module = original_module
        self.last_weights = None

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, **kwargs):
        # Force need_weights=True to get the attention map
        output, weights = self.module(query, key, value, 
                                      key_padding_mask=key_padding_mask, 
                                      need_weights=True, 
                                      attn_mask=attn_mask)
        self.last_weights = weights # Save them
        return output, weights

# --- 3. MAIN SCRIPT ---
def visualize_word_attention(model_path, data_path, latin_word, target_lang, output_dir, device):
    print(f"Visualizing Attention for: {latin_word} -> {target_lang}")
    
    # 1. Load Model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    data_artifacts = checkpoint['data_artifacts']
    
    model = PhonemeTransformer(
        num_phonemes=data_artifacts["PHONEME_VOCAB_SIZE"],
        psv_size=data_artifacts["PSV_VECTOR_SIZE"],
        d_model=config["embed_size"],
        nhead=config["nhead"],
        num_encoders=config["num_encoder_layers"],
        num_decoders=config["num_decoder_layers"],
        d_ffn=config["dim_feedforward"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 2. Inject the Probe
    # We wrap the Cross-Attention layer of the LAST decoder block
    # Structure: transformer -> decoder -> layers -> [-1] -> self_attn (decoder self) + multihead_attn (cross)
    # We want the cross-attention block where decoder attends to encoder output
    last_layer = model.transformer.decoder.layers[-1]
    cross_attn_probe = AttentionProbe(last_layer.multihead_attn)
    last_layer.multihead_attn = cross_attn_probe
    
    print("Attention Probe injected into Decoder Layer [-1] cross-attention.")
    
    # 3. Prepare Input
    # Find the ground truth in the dataset to get the correct Target Phonemes
    df = pd.read_csv(data_path)
    src_col = 'Vulgar Latin_phonemes'
    tgt_col = f'{target_lang}_phonemes'
    
    # Simple search
    row = df[df[src_col] == latin_word].iloc[0] if not df[df[src_col] == latin_word].empty else None
    
    if row is None:
        print(f"Word '{latin_word}' not found in dataset. Using dummy target for visualization.")
        target_phonemes = list(latin_word) # Fallback
    else:
        target_phonemes = list(row[tgt_col])
        print(f"Found Target: {target_phonemes}")
        
    src_phonemes = list(latin_word)
    
    # Tokenize
    vocab = data_artifacts["phoneme_to_id"]
    lang_token = data_artifacts["LANG_TOKENS"][target_lang]
    
    src_tokens = [vocab[lang_token]] + [vocab.get(p, vocab["[UNK]"]) for p in src_phonemes] + [vocab["[EOS]"]]
    tgt_tokens = [vocab["[SOS]"]] + [vocab.get(p, vocab["[UNK]"]) for p in target_phonemes] + [vocab["[EOS]"]]
    
    src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(1).to(device) # (len, 1)
    tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long).unsqueeze(1).to(device)
    tgt_input = tgt_tensor[:-1, :]  # Teacher-forcing input ([SOS] + target phonemes)
    
    # 4. Run Model
    with torch.no_grad():
        src_mask, tgt_pad_mask, tgt_mask = create_masks(
            src_tensor, tgt_input, data_artifacts["PAD_IDX"], device
        )
        model(src_tensor, tgt_input, src_mask, tgt_pad_mask, tgt_mask)
        
    # 5. Extract Weights
    # Shape: (Batch, Tgt_Len, Src_Len)
    weights = cross_attn_probe.last_weights.squeeze(0).cpu().numpy()
    
    # 6. Plot Heatmap
    plt.figure(figsize=(10, 8))
    
    # Labels
    x_labels = [lang_token] + src_phonemes + ["[EOS]"]
    y_labels = target_phonemes + ["[EOS]"]
    
    ax = sns.heatmap(weights, xticklabels=x_labels, yticklabels=y_labels, 
                     cmap="viridis", square=True, cbar_kws={'label': 'Attention Weight'})
    ax.invert_yaxis()
    
    plt.xlabel("Input (Latin)", fontsize=14)
    plt.ylabel(f"Output ({target_lang})", fontsize=14)
    plt.title(f"Cross-Attention Map\n{latin_word} $\\rightarrow$ {''.join(target_phonemes)}", fontsize=16)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"attention_{target_lang}_{latin_word}.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/processed/full_dataset.csv")
    parser.add_argument("--word", default="caballum", help="Latin phonemes (e.g., caballum)")
    parser.add_argument("--lang", default="French")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualize_word_attention(args.model, args.data, args.word, args.lang, "results/visuals", device)