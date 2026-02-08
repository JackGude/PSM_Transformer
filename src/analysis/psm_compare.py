"""Compare ground-truth vs model-predicted PSV/PSM shift matrices for one language.

This script:
- Loads a trained checkpoint and runs inference on a test CSV.
- Filters batches down to examples for a single target language.
- Aggregates:
  - ground-truth multi-hot PSV vectors into an average occurrence rate, and
  - model logits into average predicted probabilities.
- Visualizes both as sparse "shift matrices" (source phoneme x target phoneme):
  - left: rules observed in the dataset (includes noise/typos)
  - right: rules the model predicts above a confidence threshold

Output: `results/visuals/compare_matrix_<LANG>.png`.
"""

# src/compare_matrices.py
import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from src.dataset import PhonemeDataset, collate_fn
    from src.model import PhonemeTransformer, create_masks
except ImportError:
    from dataset import PhonemeDataset, collate_fn
    from model import PhonemeTransformer, create_masks


# --- 2. COMPARISON SCRIPT ---


def compare_actual_vs_predicted(
    model_path, test_data_path, output_dir, device, target_language="Spanish"
):
    print(f"Comparing Actual vs Predicted for {target_language}...")

    # Load Model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    data_artifacts = checkpoint["data_artifacts"]

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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load Data
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
    # We create a loader but we need to filter for JUST our target language in the loop
    collate_with_padding = partial(collate_fn, pad_idx=PAD_IDX)
    loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_with_padding,
    )

    # Reverse lookup for lang tokens
    id_to_phoneme = {v: k for k, v in data_artifacts["phoneme_to_id"].items()}
    target_token_str = data_artifacts["LANG_TOKENS"][target_language]

    # Accumulators
    actual_counts = np.zeros(data_artifacts["PSV_VECTOR_SIZE"])
    predicted_probs = np.zeros(data_artifacts["PSV_VECTOR_SIZE"])
    count_samples = 0

    print("Running Inference...")
    with torch.no_grad():
        for src, tgt, psm in tqdm(loader):
            src = src.to(device)
            tgt_input = tgt[:-1, :].to(device)
            src_mask, tgt_pad_mask, tgt_mask = create_masks(
                src, tgt_input, PAD_IDX, device
            )
            _, psm_logits = model(src, tgt_input, src_mask, tgt_pad_mask, tgt_mask)
            batch_probs = torch.sigmoid(psm_logits).cpu().numpy()
            batch_psm = psm.cpu().numpy()

            lang_token_ids = src[0, :].cpu().numpy()

            for i in range(batch_probs.shape[0]):
                token_str = id_to_phoneme.get(lang_token_ids[i], "UNK")

                # ONLY process the target language
                if token_str == target_token_str:
                    actual_counts += batch_psm[i]
                    predicted_probs += batch_probs[i]
                    count_samples += 1

    # Normalize
    if count_samples == 0:
        print(f"No samples found for {target_language}")
        return

    # Filter: Show rule if it appears in > 1% of words (to keep plot clean)
    avg_actual = actual_counts / count_samples
    avg_predicted = predicted_probs / count_samples

    # --- PREPARE PLOT DATA ---
    # Need to map Vector Index -> Matrix Coordinates (Src Index, Tgt Index)

    # 1. Reconstruct Phoneme List (Sorted Vowels -> Consonants)
    # Note: We need the exact same sort order as previous plots for visual consistency
    # We will infer it from the vocabulary keys
    def get_phonemes_from_shift(shift_str):
        if "->" not in shift_str:
            return None, None
        src, tgt = shift_str.split("->")
        return src, tgt

    all_shifts = list(psv_shift_to_index.keys())
    unique_phonemes = set()
    for s in all_shifts:
        src, tgt = get_phonemes_from_shift(s)
        if src:
            unique_phonemes.add(src)
            unique_phonemes.add(tgt)

    VOWELS = set(
        [
            "a",
            "e",
            "i",
            "o",
            "u",
            "y",
            "ɛ",
            "ɔ",
            "ə",
            "ɨ",
            "̃",
            "ɑ",
            "ø",
            "œ",
            "æ",
            "ʊ",
            "ɪ",
        ]
    )
    vowels_list = sorted(
        [p for p in unique_phonemes if p in VOWELS or any(v in p for v in VOWELS)]
    )
    consonants_list = sorted([p for p in unique_phonemes if p not in vowels_list])
    phonemes = vowels_list + consonants_list
    p_to_i = {p: i for i, p in enumerate(phonemes)}
    n = len(phonemes)

    # 2. Fill Matrices
    mat_actual = np.full((n, n), np.nan)
    mat_pred = np.full((n, n), np.nan)

    threshold = (
        0.01  # Only show rules the model is threshold% confident in (removes super weak noise)
    )

    for idx, rule in index_to_psv_shift.items():
        src, tgt = get_phonemes_from_shift(rule)
        if src and tgt and src in p_to_i and tgt in p_to_i:
            i, j = p_to_i[src], p_to_i[tgt]

            # Fill Actual
            if avg_actual[idx] > 0.001:  # Exists in dataset
                mat_actual[i, j] = 1

            # Fill Predicted
            if avg_predicted[idx] > threshold:
                mat_pred[i, j] = 1

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Plot 1: ACTUAL
    y, x = np.where(mat_actual == 1)
    axes[0].scatter(x, y, c="black", s=20, marker="s")
    axes[0].set_title(
        f"Actual Data ({target_language})\n(Includes Noise & Typos)", fontsize=16
    )

    # Plot 2: PREDICTED
    y, x = np.where(mat_pred == 1)
    axes[1].scatter(x, y, c="red", s=20, marker="o")
    axes[1].set_title(
        f"Model Prediction ({target_language})\n(Filtered Rules > {threshold*100:.0f}% Confidence)",
        fontsize=16,
    )

    # Styling for both
    for ax in axes:
        ax.set_xticks(range(n))
        ax.set_xticklabels(phonemes, rotation=90, fontsize=6)
        ax.set_yticks(range(n))
        ax.set_yticklabels(phonemes, fontsize=6)
        ax.xaxis.tick_top()
        ax.grid(color="#f0f0f0", linestyle=":", linewidth=0.5)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)

        # Add Red Quadrant lines
        split_idx = len(vowels_list) - 0.5
        ax.axvline(x=split_idx, color="red", linestyle="--", alpha=0.3)
        ax.axhline(y=split_idx, color="red", linestyle="--", alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"compare_matrix_{target_language}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Comparison saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/processed/test_dataset.csv")
    parser.add_argument("--lang", default="Spanish")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compare_actual_vs_predicted(
        args.model, args.data, "results/visuals", device, args.lang
    )
