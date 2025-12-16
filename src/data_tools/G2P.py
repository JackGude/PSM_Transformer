import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import torch
import warnings

# --- 1. Setup GPU Device ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU (cuda)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal (mps)")
else:
    device = torch.device("cpu")
    print("GPU not found. Running on CPU (will be very slow).")

# --- 2. Load Model ---
print("Loading model and tokenizer...")
warnings.filterwarnings("ignore", ".*Passed along max_length.*")
model = AutoModelForSeq2SeqLM.from_pretrained("charsiu/g2p_multilingual_byT5_small_100")
tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

model.to(device)
print(f"Model moved to {device}.")

# --- 3. Setup File and Language Info ---
INPUT_FOLDER = "data/unprocessed/"
INPUT_FILE = INPUT_FOLDER + "Cognates.xlsx"
INPUT_SHEET = "4 Languages"
OUTPUT_FOLDER = "data/processed/"
OUTPUT_FILE = OUTPUT_FOLDER + "phonemes_test.csv" 

COLUMNS_TO_PROCESS = {
    'Vulgar Latin': 'lat-eccl',
    'Spanish': 'spa',
    'French': 'fra',
    'Italian': 'ita',
    'Romanian': 'ron'
}

BATCH_SIZE = 512

# --- 4. Read Data ---
try:
    print(f"Reading '{INPUT_SHEET}' sheet from {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE, sheet_name=INPUT_SHEET)
    df_results = df.copy() 
    print(f"Successfully loaded {len(df)} rows.")
except Exception as e:
    print(f"ERROR reading Excel file: {e}")
    exit()

# --- 5. Process Data in Batches ---

with torch.no_grad():
    for column_name, lang_code in COLUMNS_TO_PROCESS.items():
        print(f"\nProcessing column: {column_name} (using code: {lang_code})")
        
        if column_name not in df.columns:
            print(f"Warning: Column '{column_name}' not found. Skipping.")
            continue
            
        column_data = df[column_name].fillna('')
        all_phonemes = []
        
        # Use tqdm on the range object
        for i in tqdm(range(0, len(column_data), BATCH_SIZE), desc=f"   Batches for {column_name}"):
            batch_text_list = column_data.iloc[i:i + BATCH_SIZE].tolist()
                        
            # 1. Find the indices and text of non-empty words
            non_empty_indices = []
            batch_to_process = []
            for idx, word in enumerate(batch_text_list):
                if word:
                    non_empty_indices.append(idx)
                    batch_to_process.append(f"<{lang_code}>: {word}")

            # 2. Initialize our results batch with empty strings
            final_batch_results = [""] * len(batch_text_list)
            
            # 3. Only run the model if there's something to process
            if batch_to_process:
                # Tokenize only the non-empty batch
                out = tokenizer(batch_to_process, padding=True, add_special_tokens=False, return_tensors='pt')
                out = out.to(device)
                
                # Generate predictions
                preds = model.generate(**out, num_beams=1, max_new_tokens=50) 
                
                # Decode the results
                phones_batch = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
                
                # 4. Use the indices to place results back in the correct spots
                for idx, phonemes in zip(non_empty_indices, phones_batch):
                    final_batch_results[idx] = phonemes
                        
            # 5. Add this batch's results (now in the correct order) to the full list
            all_phonemes.extend(final_batch_results)
            
        df_results[f'{column_name}_phonemes'] = all_phonemes
# --- 6. Save Results to a New Sheet ---
print(f"\nAll processing complete. Saving results to new file: {OUTPUT_FILE}...")
try:
    # Use to_csv to save the results
    df_results.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    
    print(f"Done! Data saved to {OUTPUT_FILE}.")
    print("Your original Excel file was NOT modified.")

except Exception as e:
    print("\nERROR: Could not write to CSV file.")
    print(f"Error details: {e}")