# src/create_test_set.py
import argparse
import os
import sys

import pandas as pd


def create_sample_set(input_file, test_file, train_file, num_samples):
    """
    Randomly samples rows from a full dataset to create two new files:
    1. A smaller 'test_file' for manual verification.
    2. A larger 'train_file' with all the remaining data.
    """

    # --- 1. Validate Input File ---
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        sys.exit(1)

    print(f"Reading full dataset from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {len(df)} total rows.")
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # --- 2. Check if we have enough data ---
    if len(df) < num_samples:
        print(
            f"Warning: You requested {num_samples} samples, but the file only has {len(df)} rows."
        )
        print("Will sample all available rows.")
        num_samples = len(df)

    # --- 3. Perform Random Sample ---
    print(f"Randomly sampling {num_samples} rows for the test set...")
    # random_state=42 makes sure you get the *same* random sample every time
    test_df = df.sample(n=num_samples, frac=None, random_state=42)

    # --- 4. Get the Remaining Data ---
    # We drop the rows from the original dataframe *by their index*
    train_df = df.drop(test_df.index)
    print(f"Created training set with {len(train_df)} rows.")

    # --- 5. Save Both Files ---
    try:
        print(f"Saving test set to {test_file}...")
        test_df.to_csv(test_file, index=False, encoding="utf-8")

        print(f"Saving new training set to {train_file}...")
        train_df.to_csv(train_file, index=False, encoding="utf-8")

        print(f"Your test set is ready for verification at: {test_file}")
        print(f"Your training set is ready for use at: {train_file}")

    except Exception as e:
        print(f"Error saving files: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a full dataset into a new train file and a test sample for verification."
    )

    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        help="Path to the full dataset (e.g., 'data/processed/full_dataset.csv')",
        default="data/processed/full_dataset.csv",
    )
    parser.add_argument(
        "-t",
        "--train_file",
        type=str,
        help="Path to save the new *training* set (e.g., 'data/processed/train_dataset.csv')",
        default="data/processed/train_dataset.csv",
    )
    parser.add_argument(
        "-T",
        "--test_file",
        type=str,
        help="Path to save the new *test* sample (e.g., 'data/processed/test_dataset.csv')",
        default="data/processed/test_dataset.csv",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=500,
        help="The number of random rows to sample for the test set (default: 500)",
    )

    args = parser.parse_args()

    create_sample_set(
        args.input_file, args.test_file, args.train_file, args.num_samples
    )
