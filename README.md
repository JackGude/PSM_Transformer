# PSM Transformer (Phoneme Shift Modeling)

This repository contains a linguistics / computational pipeline for modeling phonological change from (Vulgar) Latin to modern Romance languages.

At a high level:

- We start with a noisy cognate dataset (`data/unprocessed/`).
- We run G2P to turn words into phoneme strings (`data/processed/`).
- We align Latin → target language phoneme strings via Needleman–Wunsch to extract phoneme-shift rules.
- We train a multitask Transformer that:
  - translates Latin phoneme sequences into target-language phoneme sequences, and
  - predicts a sparse representation of phoneme shifts for that word (the PSV).
- We run analysis scripts to visualize drift, compare predictions vs ground truth, and inspect attention.

## Terminology: PSM vs PSV

- **PSM (Phoneme Shift Matrix)**
  - Conceptually, a dense matrix (e.g. `64 x 64`) covering *all* possible source→target phoneme substitutions.
  - Mostly empty in practice.

- **PSV (Phoneme Shift Vector)**
  - A sparse/compact representation containing only the shifts that actually occur in the dataset.
  - In this codebase, PSV is represented as:
    - a **master vocabulary** of observed shifts (strings like `"k->ʃ"`, `"a->o"`, deletions like `"x->-"`), and
    - a **dense multi-hot vector** over that vocabulary during training.
  - You can map a PSV back into a PSM by decoding each shift string and setting the corresponding matrix cell.

Note: some variables in the code use `psm_*` names even when they are actually storing the PSV-style vector.

## Repository layout

- `src/`
  - `dataset.py`: dataset + vocab building + collate function
  - `model.py`: multitask Transformer + mask helpers
  - `data_tools/`: preprocessing utilities (G2P, splitting, weighting)
  - `alignment/`: Needleman–Wunsch alignment + PSV extraction
  - `training/`: train/eval entry points
  - `analysis/`: analysis and visualization scripts

- `data/`
  - `unprocessed/`: raw dataset(s) (e.g. Excel)
  - `processed/`: processed CSVs, PSV vocabulary, and optional weight tensors

- `results/`
  - model checkpoints, plots, and analysis outputs (paths vary slightly by script/config)

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- **PyTorch install** can vary depending on whether you want CPU-only vs CUDA.
  If `pip install -r requirements.txt` does not give you the desired torch build,
  install torch first following the official instructions, then install the rest.

## How to run (end-to-end)

Most scripts are easiest to run from the repository root using `-m` so that imports like `import src.dataset` resolve correctly.

### 1) G2P: raw words → phoneme strings

Script: `src/data_tools/G2P.py`

- Reads: `data/unprocessed/Cognates.xlsx` (sheet `4 Languages`) by default
- Writes: a CSV under `data/processed/` with new `*_phonemes` columns

Run:

```bash
python -m src.data_tools.G2P
```

Important:

- The output filename is controlled by constants in the script (currently `phonemes_test.csv`).
- The alignment step below expects `data/processed/phonemes.csv` by default.
  - Either rename the output file, or edit `INPUT_FILE` in `src/alignment/N_W.py` to match.

### 2) Alignment + PSV extraction: phoneme CSV → full dataset + PSV vocabulary

Script: `src/alignment/N_W.py`

- Reads: `data/processed/phonemes.csv` (configurable)
- Writes:
  - `data/processed/psv_vocabulary.json` (master list of observed shifts)
  - `data/processed/full_dataset.csv` (adds per-language PSV columns as JSON-encoded sparse lists)

Run:

```bash
python -m src.alignment.N_W
```

### 3) Create train/test files (and a manual verification sample)

Script: `src/data_tools/split_data.py`

- Reads: `data/processed/full_dataset.csv`
- Writes:
  - `data/processed/train_dataset.csv`
  - `data/processed/test_dataset.csv`

Run (defaults shown):

```bash
python -m src.data_tools.split_data \
  --input_file data/processed/full_dataset.csv \
  --train_file data/processed/train_dataset.csv \
  --test_file data/processed/test_dataset.csv \
  --num_samples 500
```

### 4) (Optional) Compute PSV frequency weights for the auxiliary loss

Script: `src/data_tools/calculate_weights.py`

- Reads:
  - `data/processed/train_dataset.csv`
  - `data/processed/psv_vocabulary.json`
- Writes:
  - `data/processed/psv_weights.pt`

Run:

```bash
python -m src.data_tools.calculate_weights
```

### 5) Train the multitask Transformer

Script: `src/training/train.py`

- Reads (by default):
  - `data/processed/train_dataset.csv` (as the noisy training pool)
  - `data/processed/test_dataset.csv` (as the held-out test set)
  - `data/processed/psv_vocabulary.json`
  - optionally `data/processed/psv_weights.pt`
- Writes:
  - a checkpoint (default: `results/phoneme_transformer.pth`)
  - a loss plot (default: `results/loss_history.png`)

Run:

```bash
python -m src.training.train
```

Training details:

- **Translation head**: token-level phoneme prediction with `CrossEntropyLoss`.
- **Auxiliary PSV head**: multi-label prediction with `BCEWithLogitsLoss`.
  - `positive_weight` and optional `psv_weights.pt` address class imbalance.
- The saved checkpoint includes:
  - `model_state_dict`
  - `config`
  - `data_artifacts` (vocabularies, PSV shift mapping, PAD index, language tokens)

### 6) Evaluate a checkpoint on a CSV test set

Script: `src/training/test.py`

Run:

```bash
python -m src.training.test \
  --model results/phoneme_transformer.pth \
  --data data/processed/test_dataset.csv \
  --batch_size 32
```

Outputs:

- translation loss
- auxiliary loss
- micro/macro F1 for the PSV head (threshold 0.5)

## Analysis scripts (`src/analysis/`)

These scripts generally load a training checkpoint and use the saved `data_artifacts` to interpret the model.

### `analyze_model.py`: language centroids + dendrogram

- Computes per-language centroid PSV vectors from model predictions.
- Extracts “top” consonant-to-consonant shifts (strict filter).
- Produces a distance heatmap and hierarchical clustering dendrogram.

Run:

```bash
python -m src.analysis.analyze_model \
  --model results/phoneme_transformer.pth \
  --data data/processed/test_dataset.csv \
  --out results/analysis
```

### `pca_drift.py`: 2D PCA drift map (model-derived)

- PCA of language centroid vectors (Latin is an all-zero origin).
- Produces a polished 2D “drift map” plot.

Run:

```bash
python -m src.analysis.pca_drift \
  --model results/phoneme_transformer.pth \
  --data data/processed/test_dataset.csv \
  --out results/analysis
```

### `pca_drift_3d.py`: 3D PCA drift map (model-derived)

- Same centroid computation, but PCA in 3D.
- Optional rotating GIF export.

Run:

```bash
python -m src.analysis.pca_drift_3d \
  --model results/phoneme_transformer.pth \
  --data data/processed/test_dataset.csv \
  --out results/analysis

# with GIF
python -m src.analysis.pca_drift_3d \
  --model results/phoneme_transformer.pth \
  --data data/processed/test_dataset.csv \
  --out results/analysis \
  --gif
```

### `pca_ground_truth.py`: 3D PCA drift map (ground truth)

- Builds per-language vectors directly from the dataset PSV columns.
- PCA in 3D + optional rotating GIF.

Run:

```bash
python -m src.analysis.pca_ground_truth \
  --data data/processed/test_dataset.csv \
  --vocab data/processed/psv_vocabulary.json \
  --out results/visuals \
  --gif
```

### `psm_compare.py`: compare ground truth vs predicted shift matrices

- For a chosen language, compares:
  - rules observed in the dataset vs
  - rules predicted by the model above a confidence threshold.

Run:

```bash
python -m src.analysis.psm_compare \
  --model results/phoneme_transformer.pth \
  --data data/processed/test_dataset.csv \
  --lang Spanish
```

### `visualize_attention.py`: cross-attention heatmap for one word

- Injects a probe into the last decoder layer’s cross-attention.
- Produces a heatmap showing which input phonemes the decoder attends to.

Run:

```bash
python -m src.analysis.visualize_attention \
  --model results/phoneme_transformer.pth \
  --data data/processed/full_dataset.csv \
  --word caballum \
  --lang French
```

### `prove_context.py`: context sensitivity experiment

- A small probe experiment that checks whether a rule’s predicted probability
  changes depending on word position/context.
- Currently hard-coded to Romanian + rule `p->b`.

Run:

```bash
python -m src.analysis.prove_context \
  --model results/phoneme_transformer.pth \
  --data data/processed/test_dataset.csv
```

## Notes / gotchas

- **File naming consistency matters.**
  - `G2P.py` and `N_W.py` currently have different default filenames.
  - If you change one, update the other or rename outputs.

- **Import style.**
  - Many scripts assume execution via `python -m ...` from repo root.

- **Noise is expected.**
  - The dataset, G2P output, and alignments are intentionally treated as noisy;
    several analysis scripts try to filter for “high-signal” consonant shifts.
