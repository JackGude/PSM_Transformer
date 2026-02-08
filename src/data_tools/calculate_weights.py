"""Compute per-shift weights for the PSV/PSM loss based on training-set frequency.

 This script scans the sparse PSV columns (JSON-encoded shift lists) in
 `data/processed/train_dataset.csv`, counts how often each shift occurs, and
 saves a normalized weight tensor to `data/processed/psv_weights.pt`.

 `src/training/train.py` will load this tensor (if present) and pass it to
 `BCEWithLogitsLoss(weight=...)` so that common rules contribute more to the
 auxiliary loss than extremely rare/noisy ones.
 """

# src/calculate_weights.py
from collections import Counter
import json

import pandas as pd
import torch
from tqdm import tqdm

data_path = "data/processed/train_dataset.csv"

def calculate_and_save_weights():

    print("Loading data files...")
    try:
        df = pd.read_csv(data_path)
        with open("data/processed/psv_vocabulary.json", "r", encoding="utf-8") as f:
            psv_master_list = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure your data files are in data/processed/")
        return

    psv_shift_to_index = {shift: i for i, shift in enumerate(psv_master_list)}
    PSV_VECTOR_SIZE = len(psv_master_list)
    LANGUAGES = ["French", "Spanish", "Italian", "Romanian"]
    psv_cols = {lang: f"PSV_{lang}" for lang in LANGUAGES}

    print(f"Found {PSV_VECTOR_SIZE} total unique rules.")

    # 1. Count the frequency of every single rule
    rule_frequencies = Counter()
    print("Counting rule frequencies across all languages...")

    for lang in LANGUAGES:
        col = psv_cols[lang]
        for psv_sparse_str in tqdm(df[col].dropna(), desc=f"Scanning {lang}"):
            if psv_sparse_str != "[]":
                try:
                    sparse_shifts = json.loads(psv_sparse_str)
                    # We use Counter.update() to add all shifts in the list
                    rule_frequencies.update(sparse_shifts)
                except json.JSONDecodeError:
                    pass

    print("Count complete.")

    # 2. Create the weight tensor
    # We want to "prioritize the common stuff"
    # So, the weight for each rule *is* its frequency.
    # The loss for a common rule will be multiplied by a big number.
    # The loss for a rare/noisy rule will be multiplied by a small number (e.g., 1).

    weights_tensor = torch.zeros(PSV_VECTOR_SIZE, dtype=torch.float32)

    total_rules_found = 0
    for shift, index in psv_shift_to_index.items():
        freq = rule_frequencies.get(shift, 0)
        if freq > 0:
            weights_tensor[index] = float(freq)
            total_rules_found += 1

    print(f"Found frequencies for {total_rules_found} rules.")

    # 3. Normalize the weights (optional but recommended)
    # This prevents the loss from "exploding"
    # We'll scale them so the max weight is, e.g., 100
    if weights_tensor.max() > 0:
        weights_tensor = (weights_tensor / weights_tensor.max()) * 100.0

    # 4. Save the tensor
    save_path = "data/processed/psv_weights.pt"
    torch.save(weights_tensor, save_path)
    print(f"PSV weights tensor saved to {save_path}")


if __name__ == "__main__":
    calculate_and_save_weights()
