"""Probe for context sensitivity in the PSV/PSM head.

This script is a small experiment that checks whether the model's confidence in
a particular shift rule changes depending on phoneme context/position.

Current implementation:
- Loads a trained checkpoint and iterates over the test dataset.
- Filters to a single target language (currently Romanian) and a single rule
  (currently hard-coded to `p->b`).
- Collects the model's predicted probability for that rule and compares
  examples where the source phoneme (`p`) occurs word-initially vs medially.

Output is printed to stdout; the goal is qualitative evidence that the model is
not merely memorizing unigram shifts but is sensitive to surrounding context.
"""

# src/prove_context.py
import argparse
import torch
import numpy as np
import pandas as pd
from functools import partial
from torch.utils.data import DataLoader

try:
    from src.dataset import PhonemeDataset, collate_fn
    from src.model import PhonemeTransformer, create_masks
except ImportError:
    from dataset import PhonemeDataset, collate_fn
    from model import PhonemeTransformer, create_masks


def check_context_sensitivity(model_path, test_data_path, device):
    print("--- The Context Experiment ---")

    # Load Model
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

    df = pd.read_csv(test_data_path)
    psv_shift_to_index = data_artifacts["psv_shift_to_index"]

    # Find rule index
    target_rule = "p->b"
    if target_rule not in psv_shift_to_index:
        print(f"Error: Rule {target_rule} not found in model vocab.")
        return
    rule_idx = psv_shift_to_index[target_rule]

    # Setup Data
    target_lang = "Romanian"
    lang_token_str = data_artifacts["LANG_TOKENS"][target_lang]
    id_to_phoneme = {v: k for k, v in data_artifacts["phoneme_to_id"].items()}
    PAD_IDX = data_artifacts["PAD_IDX"]

    test_dataset = PhonemeDataset(
        df,
        data_artifacts["phoneme_to_id"],
        psv_shift_to_index,
        config["languages"],
        data_artifacts["LANG_TOKENS"],
    )
    collate_with_padding = partial(collate_fn, pad_idx=PAD_IDX)
    loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_with_padding,
    )

    initial_probs = []
    medial_probs = []

    print("Scanning test set...")
    with torch.no_grad():
        for src, tgt, psm in loader:
            src = src.to(device)
            tgt_input = tgt[:-1, :].to(device)

            # Check Language
            lang_id = src[0, 0].item()
            if id_to_phoneme.get(lang_id) != lang_token_str:
                continue

            # Decode Latin Word
            word_ids = src[1:-1, 0].cpu().numpy()
            word_phonemes = [id_to_phoneme.get(pid, "?") for pid in word_ids if pid != PAD_IDX]

            if 'p' not in word_phonemes:
                continue

            # Run Model
            src_mask, tgt_pad_mask, tgt_mask = create_masks(src, tgt_input, PAD_IDX, device)
            _, psm_logits = model(src, tgt_input, src_mask, tgt_pad_mask, tgt_mask)

            prob = torch.sigmoid(psm_logits)[0, rule_idx].item()

            # Context Logic
            p_indices = [j for j, x in enumerate(word_phonemes) if x == 'p']
            for p_idx in p_indices:
                if p_idx == 0:
                    initial_probs.append(prob)
                elif 0 < p_idx < len(word_phonemes)-1:
                    medial_probs.append(prob)

    print("\n=== RESULTS ===")
    print(f"Start of Word (p...): {len(initial_probs)} samples")
    print(f"Middle of Word (...p...): {len(medial_probs)} samples")

    avg_initial = np.mean(initial_probs) if initial_probs else 0
    avg_medial = np.mean(medial_probs) if medial_probs else 0

    print(f"\nConf (Start):  {avg_initial:.2%}")
    print(f"Conf (Middle): {avg_medial:.2%}")

    if avg_medial > avg_initial * 1.5:
        print("\nSUCCESS: Context sensitivity detected")
    else:
        print("\nNo context sensitivity detected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/processed/test_dataset.csv")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_context_sensitivity(args.model, args.data, device)