# src/pca_drift_3d.py
import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
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


def generate_3d_drift_map(model_path, test_data_path, output_dir, device, make_gif=False):
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

    lang_sums = {language: np.zeros(data_artifacts["PSV_VECTOR_SIZE"]) for language in languages}
    lang_counts = {language: 0 for language in languages}

    id_to_phoneme = {v: k for k, v in data_artifacts["phoneme_to_id"].items()}

    with torch.no_grad():
        for src, tgt, psm in tqdm(loader):
            src = src.to(device)
            tgt_input = tgt[:-1, :].to(device)
            src_mask, tgt_pad_mask, tgt_mask = create_masks(src, tgt_input, PAD_IDX, device)
            _, psm_logits = model(src, tgt_input, src_mask, tgt_pad_mask, tgt_mask)

            probs = torch.sigmoid(psm_logits).cpu().numpy()

            lang_token_ids = src[0, :].cpu().numpy()
            for i in range(probs.shape[0]):
                token_str = id_to_phoneme.get(lang_token_ids[i], "UNK")
                for l_name, l_tok in data_artifacts["LANG_TOKENS"].items():
                    if l_tok == token_str:
                        lang_sums[l_name] += probs[i]
                        lang_counts[l_name] += 1
                        break

    print("Running PCA (3 Components)...")
    latin_vector = np.zeros(data_artifacts["PSV_VECTOR_SIZE"])
    vectors = [latin_vector]
    labels = ["Latin (Origin)"]

    color_map = {
        'French': '#e74c3c',
        'Spanish': '#f39c12',
        'Italian': '#27ae60',
        'Romanian': '#2980b9',
        'Latin (Origin)': '#2c3e50'
    }

    for lang in languages:
        if lang_counts[lang] > 0:
            vec = lang_sums[lang] / lang_counts[lang]
            vectors.append(vec)
            labels.append(lang)

    X = np.array(vectors)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    var = pca.explained_variance_ratio_
    total_var = np.sum(var)
    print(f"Explained Variance: PC1={var[0]:.1%}, PC2={var[1]:.1%}, PC3={var[2]:.1%}")
    print(f"Total Variance Explained: {total_var:.1%}")

    sns.set_style("white")
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        X_pca[0, 0], X_pca[0, 1], X_pca[0, 2],
        c=color_map['Latin (Origin)'], s=600, marker='*',
        edgecolors='white', linewidth=2, label='Latin (Origin)', depthshade=False
    )

    for i in range(1, len(labels)):
        lang = labels[i]
        ax.scatter(
            X_pca[i, 0], X_pca[i, 1], X_pca[i, 2],
            c=color_map.get(lang, 'gray'), s=500,
            edgecolors='white', linewidth=2, label=lang, depthshade=False
        )
        ax.text(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], f" {lang}", fontsize=14, fontweight='bold')

    for i in range(1, len(labels)):
        ax.plot(
            [X_pca[i, 0], X_pca[i, 0]],
            [X_pca[i, 1], X_pca[i, 1]],
            [0, X_pca[i, 2]],
            color='gray', linestyle=':', alpha=0.5
        )

    origin = X_pca[0]
    for i in range(1, len(labels)):
        point = X_pca[i]
        ax.plot(
            [origin[0], point[0]],
            [origin[1], point[1]],
            [origin[2], point[2]],
            color=color_map.get(labels[i], 'gray'), linestyle='--', alpha=0.3
        )

    loadings = pca.components_.T
    magnitudes = np.linalg.norm(loadings, axis=1)
    sorted_indices = magnitudes.argsort()[::-1]

    VOWELS_AND_MARKERS = set([
        'a', 'e', 'i', 'o', 'u', 'y',
        'ɛ', 'ɔ', 'ə', 'ɨ', '̃', 'ɑ', 'ø', 'œ',
        'ː', '̪', '͡', 'ˈ', '.', '̠', 'ʲ'
    ])
    top_arrows = []

    for idx in sorted_indices:
        rule_name = index_to_psv_shift[idx]
        if "->" not in rule_name:
            continue
        src, tgt = rule_name.split("->")
        if tgt == "-" or src in VOWELS_AND_MARKERS or src == tgt:
            continue
        top_arrows.append(idx)
        if len(top_arrows) >= 5:
            break
        
    scale = 1.0
    label_offset_along = 0.12
    label_offset_perp = 0.08
    z_axis = np.array([0.0, 0.0, 1.0])
    for idx in top_arrows:
        rule_name = index_to_psv_shift[idx]
        v = loadings[idx] * scale
        ax.quiver(
            origin[0], origin[1], origin[2],
            v[0], v[1], v[2],
            color='#2f2f2f', linewidths=1.5, arrow_length_ratio=0.08
        )

        end_point = origin + v
        magnitude = np.linalg.norm(v)
        if magnitude == 0:
            continue
        direction = v / magnitude
        perpendicular = np.cross(direction, z_axis)
        perp_norm = np.linalg.norm(perpendicular)
        if perp_norm < 1e-6:
            perpendicular = np.cross(direction, np.array([0.0, 1.0, 0.0]))
            perp_norm = np.linalg.norm(perpendicular)
        if perp_norm > 0:
            perpendicular /= perp_norm
        offset_point = end_point + (direction * label_offset_along) + (perpendicular * label_offset_perp)
        ax.text(
            offset_point[0], offset_point[1], offset_point[2],
            rule_name,
            fontsize=13,
            fontweight='bold',
            color='#1f1f1f',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2),
        )

    ax.set_title("3D Map of Phonetic Drift", fontsize=20)
    ax.set_xlabel(f"PC1 (Variance) {var[0]:.1%}")
    ax.set_ylabel(f"PC2 (Variance) {var[1]:.1%}")
    ax.set_zlabel(f"PC3 (Variance) {var[2]:.1%}")
    ax.view_init(elev=50, azim=230)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "pca_drift_map_3d.png")
    plt.savefig(save_path)
    print(f"3D Plot saved to {save_path}")

    if make_gif:
        print("Rendering rotation GIF...")
        def update(angle):
            ax.view_init(elev=50, azim=230 + angle)
            return ax,

        frames = np.linspace(0, 360, num=450)
        gif = animation.FuncAnimation(
            fig, update, frames=frames, interval=50, blit=False
        )
        gif_path = os.path.join(output_dir, "pca_drift_map_3d.gif")
        gif.save(gif_path, writer=animation.PillowWriter(fps=45))
        print(f"GIF saved to {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/processed/test_dataset.csv")
    parser.add_argument("--out", default="results/analysis")
    parser.add_argument("--gif", action="store_true", help="Export rotating GIF in addition to PNG")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate_3d_drift_map(args.model, args.data, args.out, device, make_gif=args.gif)