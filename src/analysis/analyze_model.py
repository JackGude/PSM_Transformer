# src/analyze.py
import argparse
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from functools import partial
from tqdm import tqdm
from torch.utils.data import DataLoader

try:
    from src.dataset import PhonemeDataset, collate_fn
    from src.model import PhonemeTransformer, create_masks
except ImportError:
    from dataset import PhonemeDataset, collate_fn
    from model import PhonemeTransformer, create_masks


def analyze_model(model_path, test_data_path, output_dir, device):

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

    print(f"Loading test data from {test_data_path}...")
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

    # --- Run Inference ---
    lang_sums = {lang: np.zeros(data_artifacts["PSV_VECTOR_SIZE"]) for lang in languages}
    lang_counts = {lang: 0 for lang in languages}

    id_to_phoneme = {v: k for k, v in data_artifacts["phoneme_to_id"].items()}

    print("Running inference...")
    with torch.no_grad():
        for src, tgt, psm in tqdm(loader):
            src = src.to(device)
            tgt_input = tgt[:-1, :].to(device)

            src_mask, tgt_pad_mask, causal_mask = create_masks(
                src, tgt_input, PAD_IDX, device
            )
            _, psm_logits = model(src, tgt_input, src_mask, tgt_pad_mask, causal_mask)

            probs = torch.sigmoid(psm_logits).cpu().numpy()

            # Identify language from the first token (Token 0 is Lang Token)
            lang_token_ids = src[0, :].cpu().numpy()

            for i in range(probs.shape[0]):
                token_id = lang_token_ids[i]
                token_str = id_to_phoneme.get(token_id, "UNK")

                found_lang = None
                for lang_name, lang_token in data_artifacts["LANG_TOKENS"].items():
                    if lang_token == token_str:
                        found_lang = lang_name
                        break

                if found_lang:
                    lang_sums[found_lang] += probs[i]
                    lang_counts[found_lang] += 1

    os.makedirs(output_dir, exist_ok=True)
    language_centroids = {}

    for lang in languages:
        if lang_counts[lang] == 0:
            print(f"Warning: no samples found for {lang}; skipping centroid")
            continue
        language_centroids[lang] = lang_sums[lang] / lang_counts[lang]

    # 1. Generate "Strict Consonant Rule Book" for Slides
    results_txt = []
    print("\n=== TOP CONSONANT SHIFTS (Strict Filter) ===")

    # Define Vowels and Markers to exclude as "Source"
    # We want to find: Consonant -> Consonant (or Consonant -> New Sound)
    VOWELS_AND_MARKERS = set([
        'a', 'e', 'i', 'o', 'u', 'y',          # Standard
        'ɛ', 'ɔ', 'ə', 'ɨ', '̃', 'ɑ', 'ø', 'œ', # IPA Vowels
        'ː', '̪', '͡', 'ˈ', '.', '̠', 'ʲ'        # Markers
    ])

    for lang in languages:
        if lang not in language_centroids:
            continue
        centroid = language_centroids[lang]

        # Look deeper (top 150) because we are filtering aggressively
        top_indices = centroid.argsort()[-150:][::-1]

        header = f"\nLanguage: {lang}"
        print(header)
        results_txt.append(header)

        count = 0
        for idx in top_indices:
            rule_name = index_to_psv_shift[idx]
            prob = centroid[idx]

            # Parse the rule (e.g., "k->s")
            if "->" not in rule_name:
                continue

            src, tgt = rule_name.split("->")

            # --- THE FILTER ---
            # 1. Reject if Source is a Vowel or Marker (we want Consonants)
            if src in VOWELS_AND_MARKERS:
                continue

            # 2. Reject if Target is Deletion (we want Transformations)
            if tgt == "-":
                continue

            # 3. Reject Identity (e.g., p->p)
            if src == tgt:
                continue
            # ------------------

            line = f"  {count+1:02d}. {rule_name:<10} (Freq: {prob:.4f})"
            print(line)
            results_txt.append(line)

            count += 1
            if count >= 10: # Just get the top 10 "Gold Standard" rules
                break

    with open(os.path.join(output_dir, "top_shifts.txt"), "w", encoding='utf-8') as file:
        file.write("\n".join(results_txt))

    # 2. Generate Dendrogram
    print("\nGenerating Dendrogram...")
    valid_languages = [language for language in languages if language in language_centroids]
    if len(valid_languages) < 2:
        print("Not enough language centroids to build dendrogram; skipping heatmap/dendrogram generation.")
        return

    centroid_matrix = np.array([language_centroids[language] for language in valid_languages])

    from sklearn.metrics.pairwise import euclidean_distances
    dist_matrix = euclidean_distances(centroid_matrix)

    # Heatmap
    df_dist = pd.DataFrame(dist_matrix, index=valid_languages, columns=valid_languages)

    sns.clustermap(df_dist, method='ward', annot=True, fmt=".2f", cmap="viridis_r")
    plt.savefig(os.path.join(output_dir, "heatmap.png"))
    plt.close()

    # Dendrogram
    condensed_dist = dist_matrix[np.triu_indices(len(valid_languages), k=1)]
    linkage_matrix = sch.linkage(condensed_dist, method='ward')
    plt.figure(figsize=(10, 5))
    sch.dendrogram(linkage_matrix, labels=valid_languages, leaf_rotation=0, leaf_font_size=12)

    plt.title("Model Derived: Phonetic Drift")
    plt.ylabel("Euclidean Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dendrogram.png"))
    plt.close()

    print(f"Done! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/processed/test_dataset.csv")
    parser.add_argument("--out", default="results/analysis")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyze_model(args.model, args.data, args.out, device)