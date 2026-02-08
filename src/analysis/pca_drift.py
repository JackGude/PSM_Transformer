"""2D PCA visualization of model-derived phonetic drift.

This script:
- Loads a trained checkpoint and runs inference on a test CSV.
- Aggregates the model's PSV/PSM predictions into per-language mean vectors
  (centroids), with an all-zero vector used as the Latin origin.
- Runs PCA (2 components) on the centroid vectors and produces a "drift map"
  plot showing languages positioned in the learned PSV space.
- Overlays a small number of salient phoneme-shift rules as arrows, chosen by
  large PCA loading magnitude and filtered to remove vowels/markers/deletions.

The main output is `pca_drift_map_2d.png` written under `--out`.
"""

# src/pca_drift_pretty.py
import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from src.dataset import PhonemeDataset, collate_fn
    from src.model import PhonemeTransformer, create_masks
except ImportError:
    from dataset import PhonemeDataset, collate_fn
    from model import PhonemeTransformer, create_masks

def generate_drift_map(model_path, test_data_path, output_dir, device):
    print(f"Loading model from {model_path}...")
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

    print("Running inference...")
    df_test = pd.read_csv(test_data_path)
    languages = config["languages"]
    psv_shift_to_index = data_artifacts["psv_shift_to_index"]
    index_to_psv_shift = {v: k for k, v in psv_shift_to_index.items()}
    PAD_IDX = data_artifacts["PAD_IDX"]
    
    test_dataset = PhonemeDataset(
        df_test,
        data_artifacts["phoneme_to_id"],
        psv_shift_to_index,
        languages,
        data_artifacts["LANG_TOKENS"],
    )
    collate_with_padding = partial(collate_fn, pad_idx=PAD_IDX)
    loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_with_padding,
    )

    language_sums = {lang: np.zeros(data_artifacts["PSV_VECTOR_SIZE"]) for lang in languages}
    language_counts = {lang: 0 for lang in languages}
    id_to_phoneme = {v: k for k, v in data_artifacts["phoneme_to_id"].items()}

    with torch.no_grad():
        for src, tgt, psv in tqdm(loader):
            src = src.to(device)
            tgt_input = tgt[:-1, :].to(device)
            src_mask, tgt_pad_mask, tgt_mask = create_masks(src, tgt_input, PAD_IDX, device)
            _, psv_logits = model(src, tgt_input, src_mask, tgt_pad_mask, tgt_mask)

            probs = torch.sigmoid(psv_logits).cpu().numpy()
            
            lang_token_ids = src[0, :].cpu().numpy()
            for i in range(probs.shape[0]):
                token_str = id_to_phoneme.get(lang_token_ids[i], "UNK")
                for l_name, l_tok in data_artifacts["LANG_TOKENS"].items():
                    if l_tok == token_str:
                        language_sums[l_name] += probs[i]
                        language_counts[l_name] += 1
                        break

    # --- 3. PREPARE PCA DATA ---
    print("Running PCA...")
    latin_vector = np.zeros(data_artifacts["PSV_VECTOR_SIZE"])
    vectors = [latin_vector]
    labels = ["Latin (Origin)"]
    
    # Prettier Colors
    color_map = {
        'French': '#e74c3c',   # Red
        'Spanish': '#f39c12',  # Orange
        'Italian': '#27ae60',  # Green
        'Romanian': '#2980b9', # Blue
        'Latin (Origin)': '#2c3e50' # Dark Blue/Black
    }
    
    for lang in languages:
        if language_counts[lang] > 0:
            vec = language_sums[lang] / language_counts[lang]
            vectors.append(vec)
            labels.append(lang)

    X = np.array(vectors)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # --- 4. PRETTY PLOTTING ---
    sns.set_style("whitegrid") # Clean background
    plt.figure(figsize=(16, 12)) # 4K Ready Size
    
    # Plot Origin separately (Star shape)
    plt.scatter(X_pca[0, 0], X_pca[0, 1], c=color_map['Latin (Origin)'], s=600, marker='*', 
                edgecolors='white', linewidth=2, label='Latin (Origin)', zorder=10)

    # Plot Languages (Circles)
    for i in range(1, len(labels)):
        lang = labels[i]
        plt.scatter(X_pca[i, 0], X_pca[i, 1], c=color_map.get(lang, 'gray'), s=500, 
                    edgecolors='white', linewidth=2, label=lang, zorder=10)
        
        # Add labels with better positioning
        offset_y = 0.05 if X_pca[i, 1] > 0 else -0.05
        plt.text(X_pca[i, 0], X_pca[i, 1] + offset_y, lang, fontsize=18, fontweight='bold', 
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

    # Draw faint connection lines to origin
    origin_x, origin_y = X_pca[0, 0], X_pca[0, 1]
    for i in range(1, len(labels)):
        lang = labels[i]
        plt.plot([origin_x, X_pca[i, 0]], [origin_y, X_pca[i, 1]], 
                 color=color_map.get(lang, 'gray'), linestyle='--', alpha=0.3, zorder=1)

    # --- 5. SMART ARROW SELECTION ---
    loadings = pca.components_.T
    magnitudes = np.linalg.norm(loadings, axis=1)
    
    # Sort by magnitude (strength)
    sorted_indices = magnitudes.argsort()[::-1]
    
    # Filter for "Interesting" Rules Only
    # Exclude Vowels, Deletions, Markers
    vowels_and_markers = set([
        'a', 'e', 'i', 'o', 'u', 'y',          # Standard
        'ɛ', 'ɔ', 'ə', 'ɨ', '̃', 'ɑ', 'ø', 'œ', # IPA Vowels
        'ː', '̪', '͡', 'ˈ', '.', '̠', 'ʲ'        # Markers
    ])
    
    top_arrows = []
    
    for idx in sorted_indices:
        rule_name = index_to_psv_shift[idx]
        if "->" not in rule_name:
            continue
        src, tgt = rule_name.split("->")

        if tgt == "-":
            continue
        if src in vowels_and_markers:
            continue
        if src == tgt:
            continue

        top_arrows.append(idx)
        if len(top_arrows) >= 5:
            break

    # Plot Arrows
    scale_factor = 1.3 # Make arrows big enough to see
    
    for idx in top_arrows:
        rule_name = index_to_psv_shift[idx]
        lx, ly = loadings[idx, 0] * scale_factor, loadings[idx, 1] * scale_factor

        # Color arrows gray/black to distinguish from languages
        plt.arrow(origin_x, origin_y, lx, ly, color='#555555', alpha=0.8, width=0.003, head_width=0.02, zorder=5)

        # Offset label away from the arrow tip for legibility
        end_x, end_y = origin_x + lx, origin_y + ly
        direction = np.array([lx, ly])
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        unit_dir = direction / norm
        perpendicular = np.array([-unit_dir[1], unit_dir[0]])
        offset_point = np.array([end_x, end_y]) + (unit_dir * 0.08) + (perpendicular * 0.05)

        plt.text(
            offset_point[0],
            offset_point[1],
            rule_name,
            color='#2c3e50',
            fontsize=14,
            fontweight='bold',
            style='italic',
            ha='center',
            va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2),
        )

    # Final Touches
    plt.title("Map of Phonetic Drift", fontsize=24, pad=20)
    plt.xlabel(f"PC1 (Variance: {pca.explained_variance_ratio_[0]:.1%})", fontsize=16)
    plt.ylabel(f"PC2 (Variance: {pca.explained_variance_ratio_[1]:.1%})", fontsize=16)
    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        fontsize=14,
        frameon=True,
        framealpha=1,
        borderaxespad=0,
    )
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "pca_drift_map_2d.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Pretty PCA Plot saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/processed/test_dataset.csv")
    parser.add_argument("--out", default="results/analysis")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate_drift_map(args.model, args.data, args.out, device)