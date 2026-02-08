"""Evaluate a saved multitask model checkpoint on a test CSV.

 This script:
 - Loads a checkpoint produced by `src/training/train.py`.
 - Reconstructs the `PhonemeTransformer` using the saved `config` and
   `data_artifacts` (vocabularies, PSV mapping, PAD index).
 - Runs a forward pass over the provided test dataset and reports:
   - Translation loss (cross-entropy over phoneme tokens).
   - PSV/PSM loss (BCEWithLogits over the dense shift vector).
   - Micro/macro F1 for the binary PSV/PSM predictions (thresholded at 0.5).

 It is intended as a lightweight CLI sanity-check / reporting tool, separate
 from the richer visualization scripts in `src/analysis/`.
 """

# src/test.py
import argparse
from functools import partial
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
from torch.amp import autocast
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Import from our custom files ---
try:
    from ..model import PhonemeTransformer, create_masks
    from ..dataset import PhonemeDataset, collate_fn
except ImportError:
    from src.model import PhonemeTransformer, create_masks
    from src.dataset import PhonemeDataset, collate_fn


def run_evaluation(model_path, test_data_path, device, batch_size):
    """
    Loads a model checkpoint and evaluates it on a specified test dataset.
    """

    # --- 1. Load Checkpoint ---
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    if not os.path.exists(test_data_path):
        print(f"Error: Test data file not found at {test_data_path}")
        sys.exit(1)

    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract the saved config and artifacts
    config = checkpoint["config"]
    data_artifacts = checkpoint["data_artifacts"]

    # --- 2. Re-build Model ---
    print("Re-building model structure...")
    try:
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
    except KeyError as e:
        print(f"Error: Hyperparameter {e} not found in saved config.")
        sys.exit(1)

    # Load the trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully.")

    # --- 3. Load Test Data ---
    print(f"Loading test data from {test_data_path}...")
    df_test = pd.read_csv(test_data_path)

    # Use the vocabularies *from the checkpoint*
    phoneme_to_id = data_artifacts["phoneme_to_id"]
    psv_shift_to_index = data_artifacts["psv_shift_to_index"]
    PAD_IDX = data_artifacts["PAD_IDX"]
    LANG_TOKENS = data_artifacts["LANG_TOKENS"]
    LANGUAGES = config["languages"]

    test_dataset = PhonemeDataset(
        df_test, phoneme_to_id, psv_shift_to_index, LANGUAGES, LANG_TOKENS
    )
    collate_with_padding = partial(collate_fn, pad_idx=PAD_IDX)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_with_padding,
    )

    print(f"Loaded {len(df_test)} test rows, creating {len(test_loader)} batches.")

    # --- 4. Initialize Loss Functions ---
    translation_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    pos_weight = torch.tensor([config["positive_weight"]], device=device)
    rules_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- 5. Run Evaluation Loop ---
    test_loss_trans = 0.0
    test_loss_rules = 0.0
    all_psv_preds = []
    all_psv_targets = []
    use_amp = torch.cuda.is_available() or torch.backends.mps.is_available()

    with torch.no_grad():
        for src, tgt, psv in tqdm(test_loader, desc="[Testing]"):
            src, tgt, psv = src.to(device), tgt.to(device), psv.to(device)

            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]

            src_mask, tgt_pad_mask, causal_mask = create_masks(
                src, tgt_input, PAD_IDX, device
            )

            with autocast(device_type=device.type, enabled=use_amp):
                trans_logits, psv_logits = model(
                    src,
                    tgt_input,
                    src_padding_mask=src_mask,
                    tgt_padding_mask=tgt_pad_mask,
                    tgt_mask=causal_mask,
                )

                loss_trans_logits = trans_logits.reshape(
                    -1, data_artifacts["PHONEME_VOCAB_SIZE"]
                )
                loss_trans_target = tgt_output.transpose(0, 1).reshape(-1)
                loss_trans = translation_loss_fn(loss_trans_logits, loss_trans_target)
                loss_rules = rules_loss_fn(psv_logits, psv)

            test_loss_trans += loss_trans.item()
            test_loss_rules += loss_rules.item()

            preds = torch.sigmoid(psv_logits)
            binary_preds = (preds > 0.5).cpu().numpy()
            all_psv_preds.append(binary_preds)
            all_psv_targets.append(psv.cpu().numpy().astype(int))

    # --- 6. Calculate Final Scores ---
    avg_test_trans_loss = test_loss_trans / len(test_loader)
    avg_test_rules_loss = test_loss_rules / len(test_loader)

    all_psv_preds = np.concatenate(all_psv_preds, axis=0)
    all_psv_targets = np.concatenate(all_psv_targets, axis=0)

    f1_micro = f1_score(
        all_psv_targets, all_psv_preds, average="micro", zero_division=0
    )
    f1_macro = f1_score(
        all_psv_targets, all_psv_preds, average="macro", zero_division=0
    )

    print("\n--- FINAL TEST RESULTS ---")
    print(f"  Model: {model_path}")
    print(f"  Test Set: {test_data_path}")
    print("---------------------------------")
    print(f"  Translation Loss:    {avg_test_trans_loss:.4f}")
    print(f"  Rules (PSV) Loss:    {avg_test_rules_loss:.4f}")
    print(f"  PSV F1 Score (Micro):  {f1_micro:.4f}")
    print(f"  PSV F1 Score (Macro):  {f1_macro:.4f}")
    print("---------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PhonemeTransformer model on a test set."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the saved .pth checkpoint file (e.g., 'results/models/phoneme_transformer.pth')",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the test data .csv file (e.g., 'data/gold_standard/gold_test_set.csv')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)",
    )

    args = parser.parse_args()

    # Auto-detect device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    run_evaluation(args.model, args.data, device, args.batch_size)
