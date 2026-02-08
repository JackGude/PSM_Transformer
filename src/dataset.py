"""Dataset and vocabulary utilities for the phoneme/PSV multitask Transformer.

 This module defines:
 - `PhonemeDataset`: yields per-(word, target-language) training examples.
   - Encoder input: Latin phoneme sequence prefixed with a language token and
     terminated with `[EOS]`.
   - Decoder input: target-language phoneme sequence in teacher-forcing format
     (`[SOS] ... [EOS]`).
   - Auxiliary target: a dense multi-hot PSV/PSM vector built from a sparse JSON
     list of phoneme-shift rules stored in the CSV (e.g. `["k->ʃ", "a->o"]`).
 - `collate_fn`: pads variable-length sequences and stacks PSV vectors.
 - `build_vocabularies`: builds the phoneme/token -> id mapping from a DataFrame.

 The training scripts in `src/training/` use these utilities to construct
 DataLoaders and consistently encode phoneme sequences and rule vectors.
 """

import torch
import pandas as pd
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Define Special Tokens
PAD_TOKEN = "[PAD]" # Padding for batches
SOS_TOKEN = "[SOS]" # "Start of Sequence" for the decoder
EOS_TOKEN = "[EOS]" # "End of Sequence" for both encoder/decoder
UNK_TOKEN = "[UNK]" # For any unknown phonemes

class PhonemeDataset(Dataset):
    def __init__(self, dataframe, phoneme_vocab, psv_vocab_map, languages, lang_tokens):
        """
        Initializes the dataset.
        - dataframe: The pandas DataFrame (e.g., your training data)
        - phoneme_vocab: The mapping from phoneme -> ID (e.g., {'a': 5, 'b': 6})
        - psv_vocab_map: The mapping from shift -> ID (e.g., {'k->ʃ': 120})
        - languages: A list of language names (e.g., ['French', 'Spanish'])
        - lang_tokens: A dict of language -> token (e.g., {'French': '[FR]'})
        """
        self.dataframe = dataframe
        self.phoneme_vocab = phoneme_vocab
        self.psv_vocab_map = psv_vocab_map
        self.psv_vector_size = len(psv_vocab_map)
        self.languages = languages
        self.lang_tokens = lang_tokens
        
        # Define the column names we'll be pulling from
        self.src_col = 'Vulgar Latin_phonemes'
        self.tgt_cols = {lang: f"{lang}_phonemes" for lang in languages}
        self.psv_cols = {lang: f"PSV_{lang}" for lang in languages}

    def __len__(self):
        """
        Returns the total number of items in the dataset.
        Since each row has 4 languages, the total is rows * 4.
        """
        return len(self.dataframe) * len(self.languages)

    def __getitem__(self, index):
        """
        Gets a single, complete training example (one language pair)
        """
        
        # Figure out which row and which language this index is for.
        lang_index = index % len(self.languages)
        row_index = index // len(self.languages)
        
        language = self.languages[lang_index]
        lang_token = self.lang_tokens[language]
        
        row = self.dataframe.iloc[row_index]
        
        # Get the phoneme strings, or an empty list if data is missing (NaN)
        src_phonemes = list(row[self.src_col]) if pd.notna(row[self.src_col]) else []
        tgt_phonemes = list(row[self.tgt_cols[language]]) if pd.notna(row[self.tgt_cols[language]]) else []

        # Source (Encoder Input): [LANG_TOKEN] + phonemes + [EOS]
        src_tokens = [self.phoneme_vocab[lang_token]] + \
                     [self.phoneme_vocab.get(p, self.phoneme_vocab[UNK_TOKEN]) for p in src_phonemes] + \
                     [self.phoneme_vocab[EOS_TOKEN]]
        
        # Target (Decoder Input): [SOS] + phonemes + [EOS]
        tgt_tokens = [self.phoneme_vocab[SOS_TOKEN]] + \
                     [self.phoneme_vocab.get(p, self.phoneme_vocab[UNK_TOKEN]) for p in tgt_phonemes] + \
                     [self.phoneme_vocab[EOS_TOKEN]]

        # Create the full, dense PSM vector
        psm_vector = torch.zeros(self.psv_vector_size, dtype=torch.float32)
        
        # Load the sparse list of shifts (e.g., '["k->ʃ", "a->o"]')
        psv_sparse_str = row[self.psv_cols[language]]
        
        if pd.notna(psv_sparse_str) and psv_sparse_str != "[]":
            try:
                # Convert the string back into a Python list
                sparse_shifts = json.loads(psv_sparse_str)
                for shift in sparse_shifts:
                    if shift in self.psv_vocab_map:
                        # Find the index for this shift in our master "dictionary"
                        idx = self.psv_vocab_map[shift]
                        # Set that position to 1.0 in our 10,000-long vector
                        psm_vector[idx] = 1.0 
            except json.JSONDecodeError:
                # Handle cases where the data might be corrupted
                pass 

        return (
            torch.tensor(src_tokens, dtype=torch.long),
            torch.tensor(tgt_tokens, dtype=torch.long),
            psm_vector
        )

def collate_fn(batch, pad_idx):
    """
    Takes a list of (src, tgt, psm) tuples and combines them into a batch.
    - src and tgt sequences are padded to the same length.
    - psm vectors are stacked.
    """
    src_batch, tgt_batch, psm_batch = [], [], []
    
    # Unzip the batch
    for src_sample, tgt_sample, psm_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
        psm_batch.append(psm_sample)

    # Pad sequences to the max length *in this batch*
    # `batch_first=True` means the output shape is (batch_size, seq_len)
    src_padded = pad_sequence(src_batch, batch_first=False, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=False, padding_value=pad_idx)
    
    # PSM vectors are all the same length, so just stack them
    # into a (batch_size, psv_vector_size) tensor
    psm_stacked = torch.stack(psm_batch)
    
    return src_padded, tgt_padded, psm_stacked

def build_vocabularies(dataframe, languages, lang_tokens):
    """
    Creates the phoneme-to-ID mapping from the dataset.
    """
    print("Building phoneme vocabulary...")
    phoneme_set = set()
    
    # Get all unique phonemes from all relevant columns
    cols_to_scan = ['Vulgar Latin_phonemes'] + [f"{lang}_phonemes" for lang in languages]
    for col in cols_to_scan:
        for item in dataframe[col].dropna():
            phoneme_set.update(list(item))
            
    # Create the final vocab list, adding special tokens at the beginning
    special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + list(lang_tokens.values())
    phoneme_list = special_tokens + sorted(list(phoneme_set))
    
    # Create the mapping dictionary
    phoneme_to_id = {phoneme: i for i, phoneme in enumerate(phoneme_list)}
    
    print(f"Phoneme vocabulary size: {len(phoneme_to_id)}")
    
    return phoneme_to_id