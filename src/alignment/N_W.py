"""Needleman–Wunsch alignment + PSV vocabulary/dataset generation.
 
 This file serves two roles:
 
 - Library utilities:
   - `needleman_wunsch`: global sequence alignment for two phoneme sequences.
   - `clean_phoneme_string`: converts a stored phoneme string into a list.
 
 - Standalone preprocessing script:
   - Reads `data/processed/phonemes.csv` (output of `src/data_tools/G2P.py`).
   - Aligns Vulgar Latin phonemes to each target language via Needleman–Wunsch.
   - Extracts per-word phoneme-shift rules (e.g. `k->ʃ`, deletions as `x->-`).
   - Writes:
     - `data/processed/psv_vocabulary.json`: master list of unique shifts.
     - `data/processed/full_dataset.csv`: original rows plus per-language PSV
       columns containing JSON-encoded sparse shift lists.
 
 The resulting `full_dataset.csv` is the main input to later splitting,
 weighting, training, and analysis scripts.
 """

import json

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- 1. Needleman-Wunsch (N-W) Algorithm Implementation ---

def needleman_wunsch(seq1, seq2, match_score=1, mismatch_score=-1, gap_score=-1):
    """
    Performs the Needleman-Wunsch alignment on two sequences (lists of phonemes).
    Returns the aligned sequences and the list of (phoneme1, phoneme2) pairs.
    """
    
    # Initialize the scoring matrix
    n = len(seq1)
    m = len(seq2)
    score_matrix = np.zeros((n + 1, m + 1))
    
    # Initialize the traceback matrix (to reconstruct the path)
    traceback_matrix = np.zeros((n + 1, m + 1), dtype=int)
    
    for i in range(1, n + 1):
        score_matrix[i][0] = score_matrix[i-1][0] + gap_score
        traceback_matrix[i][0] = 2 # From top
    for j in range(1, m + 1):
        score_matrix[0][j] = score_matrix[0][j-1] + gap_score
        traceback_matrix[0][j] = 3 # From left

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i-1] == seq2[j-1]:
                diag_score = score_matrix[i-1][j-1] + match_score
            else:
                diag_score = score_matrix[i-1][j-1] + mismatch_score
            
            up_score = score_matrix[i-1][j] + gap_score
            left_score = score_matrix[i][j-1] + gap_score
            
            scores = [diag_score, up_score, left_score]
            max_score = max(scores)
            score_matrix[i][j] = max_score
            traceback_matrix[i][j] = np.argmax(scores) + 1 
            
    # --- Traceback ---
    aligned_seq1 = []
    aligned_seq2 = []
    shifts = []
    
    i, j = n, m
    while i > 0 or j > 0:
        move = traceback_matrix[i][j]
        
        if move == 1: # Diagonal
            ph1 = seq1[i-1]
            ph2 = seq2[j-1]
            aligned_seq1.insert(0, ph1)
            aligned_seq2.insert(0, ph2)
            if ph1 != ph2:
                shifts.append(f"{ph1}->{ph2}")
            i -= 1
            j -= 1
        elif move == 2: # From top
            ph1 = seq1[i-1]
            ph2 = '-'
            aligned_seq1.insert(0, ph1)
            aligned_seq2.insert(0, ph2)
            shifts.append(f"{ph1}->{ph2}")
            i -= 1
        elif move == 3: # From left
            ph1 = '-'
            ph2 = seq2[j-1]
            aligned_seq1.insert(0, ph1)
            aligned_seq2.insert(0, ph2)
            shifts.append(f"{ph1}->{ph2}")
            j -= 1
        else: 
            break
            
    return aligned_seq1, aligned_seq2, shifts

def clean_phoneme_string(ps):
    """
    Helper function to turn a phoneme string into a list of phonemes.
    For the 'charsiu' G2P model, this just means splitting the string 
    into a list of its characters.
    """
    if not isinstance(ps, str) or pd.isna(ps):
        return []
    
    # Simply convert the string to a list of its characters
    return list(ps)


# --- 2. Setup File and Language Info ---
# NOTE: I am assuming your G2P output file is named "output_phonemes.csv"
DATA_DIRECTORY = "data/processed/"
INPUT_FILE = DATA_DIRECTORY + "phonemes.csv" 
OUTPUT_FILE = DATA_DIRECTORY + "full_dataset.csv"
OUTPUT_VOCAB_FILE = DATA_DIRECTORY + "psv_vocabulary.json" 

SOURCE_COLUMN = 'Vulgar Latin_phonemes'
TARGET_COLUMNS = [
    'French_phonemes',
    'Spanish_phonemes',
    'Italian_phonemes',
    'Romanian_phonemes'
]

# --- 3. Read Data ---
try:
    print(f"Reading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Successfully loaded {len(df)} rows.")
except FileNotFoundError:
    print(f"ERROR: Input file not found at {INPUT_FILE}")
    exit()

# --- 4. First Pass: Discover All Unique Shifts ---
print("Starting Pass 1: Discovering all unique phoneme shifts...")

# Import and initialize tqdm.pandas

tqdm.pandas()

# This helper function will be "applied" to every row
def get_shifts_from_row(row, source_col, target_col):
    """
    Runs N-W on a single row of the DataFrame.
    """
    # Clean the phoneme strings into lists
    seq1 = clean_phoneme_string(row[source_col])
    seq2 = clean_phoneme_string(row[target_col])
    
    if not seq1 or not seq2:
        return []
        
    _, _, shifts = needleman_wunsch(seq1, seq2)
    return shifts

all_shifts_vocabulary = set()
all_word_shifts_map = {} 

for target_col in TARGET_COLUMNS:
    print(f"  Aligning {SOURCE_COLUMN} -> {target_col}...")
    
    # Use .progress_apply() instead of .iterrows()
    # This runs the function on every row (axis=1) and shows a progress bar
    shift_lists_for_lang = df.progress_apply(
        get_shifts_from_row, 
        args=(SOURCE_COLUMN, target_col), 
        axis=1
    )
    
    # Now that we have all the lists, update the master vocabulary
    for shifts in shift_lists_for_lang:
        all_shifts_vocabulary.update(shifts)
    
    # Store the results (as a list)
    all_word_shifts_map[target_col] = shift_lists_for_lang.tolist()

# --- 5. Create Master Shift List and Save ---
print("\nPass 1 complete. Creating master PSV list...")

master_shift_list = sorted(list(all_shifts_vocabulary))

print(f"Found {len(master_shift_list)} unique phoneme shifts.")

with open(OUTPUT_VOCAB_FILE, 'w', encoding='utf-8') as f:
    json.dump(master_shift_list, f, ensure_ascii=False, indent=2)
print(f"PSV vocabulary saved to {OUTPUT_VOCAB_FILE}")


print("\nAssembling final DataFrame with 'sparse' PSV lists...")

for target_col in TARGET_COLUMNS:
    new_col_name = f"PSV_{target_col.replace('_phonemes', '')}"
    shifts_for_this_language = all_word_shifts_map[target_col]
    
    # Save the list as a JSON string
    df[new_col_name] = [json.dumps(word_shifts) for word_shifts in shifts_for_this_language]

# --- 6. Save Final CSV File ---
print(f"\nAll processing complete. Saving results to {OUTPUT_FILE}...")
try:
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    
    print(f"Done! Data saved to {OUTPUT_FILE}.")
    print("\nExample of the first 5 rows (PSV columns should now be correct):")
    print(df.head())

except Exception as e:
    print("\nERROR: Could not write to CSV file.")
    print(f"Error details: {e}")