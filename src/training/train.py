"""Train the multitask phoneme Transformer (translation + PSV/PSM prediction).
 
 This script is the main training entry point for the project.
 
 Responsibilities:
 - Loads the processed training CSV and PSV vocabulary from `data/processed/`.
 - Builds the phoneme token vocabulary (including per-language tokens).
 - Constructs `DataLoader`s that treat each (Latin word, target language) pair as
   a separate training example.
 - Trains a `PhonemeTransformer` with two losses:
   - Translation loss: `CrossEntropyLoss` over decoder phoneme tokens.
   - PSV/PSM loss: `BCEWithLogitsLoss` over a dense multi-label shift vector.
 - Saves the best checkpoint (by validation loss) to `CONFIG["model_save_path"]`.
 
 Checkpoint format (used by `src/training/test.py` and `src/analysis/*`):
 - `model_state_dict`
 - `config` (hyperparameters + language list)
 - `data_artifacts` (vocabularies, PAD index, PSV shift mapping, etc.)
 """

from functools import partial
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
from torch.amp import GradScaler, autocast
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Import from custom files ---
try:
    from src.dataset import PAD_TOKEN, PhonemeDataset, build_vocabularies, collate_fn
    from src.model import PhonemeTransformer, create_masks
except ImportError:
    print("ERROR: Could not import from dataset.py or model.py.")
    print("Please make sure those files are in the same directory as train.py")
    exit()

# --- 1. Configuration Dictionary ---
CONFIG = {
    # --- File Paths ---
    "data_dir": "data/processed",
    "training_data_file": "train_dataset.csv",
    "test_data_file": "test_dataset.csv",
    "psv_vocab_file": "psv_vocabulary.json",
    "model_save_path": "results/phoneme_transformer.pth",
    "plot_save_path": "results/loss_history.png",
    # --- Model Hyperparameters ---
    "embed_size": 512,
    "nhead": 4,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "dim_feedforward": 1024,
    "dropout": 0.35,
    "weight_decay": 2e-4,
    # --- Training Hyperparameters ---
    "early_stopping_patience": 30,
    "learning_rate": 0.00025,
    "batch_size": 64,
    "num_epochs": 250,
    "psv_loss_lambda": 10.0,
    "positive_weight": 10.0,
    # --- Data & Special Tokens ---
    "languages": ["French", "Spanish", "Italian", "Romanian"],
    "val_split_percent": 0.1,  # 10% of noisy data for validation
}

# --- 2. Helper Functions ---


def setup_device_and_amp():
    """Sets up the device and AMP."""
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if torch.cuda.is_available():
        print("Using device: NVIDIA GPU (cuda)")
        use_amp = True
        print("Automatic Mixed Precision (AMP) enabled.")
    elif torch.backends.mps.is_available():
        print("Using device: Apple Metal (mps)")
        use_amp = True  # AMP can be used on MPS
        print("Automatic Mixed Precision (AMP) enabled for MPS.")
    else:
        print("Using device: CPU")
        use_amp = False
        print("AMP disabled (requires GPU).")
    return device, use_amp


def load_data_and_vocabs(config):
    """Loads all data and vocabularies from disk."""
    print("Loading data and vocabularies...")
    try:
        df = pd.read_csv(config["data_dir"] + "/" + config["training_data_file"])
    except FileNotFoundError:
        print(
            f"ERROR: Could not find {config['data_dir'] + "/" + config['training_data_file']}."
        )
        return None

    with open(
        config["data_dir"] + "/" + config["psv_vocab_file"], "r", encoding="utf-8"
    ) as f:
        psv_master_list = json.load(f)
    psv_shift_to_index = {shift: i for i, shift in enumerate(psv_master_list)}

    lang_tokens = {lang: f"[{lang.upper()}]" for lang in config["languages"]}
    phoneme_to_id = build_vocabularies(df, config["languages"], lang_tokens)

    print(f"Loaded {len(df)} total rows.")
    print(f"Loaded {len(psv_master_list)} unique phoneme shifts.")

    # Return a dictionary of all our data artifacts
    return {
        "df": df,
        "psv_shift_to_index": psv_shift_to_index,
        "phoneme_to_id": phoneme_to_id,
        "PHONEME_VOCAB_SIZE": len(phoneme_to_id),
        "PSV_VECTOR_SIZE": len(psv_master_list),
        "PAD_IDX": phoneme_to_id[PAD_TOKEN],
        "LANG_TOKENS": lang_tokens,
    }


def create_dataloaders(config, data_artifacts):
    """Splits data and creates train, val, and test DataLoaders."""

    # --- 1. Get artifacts and vocab ---
    phoneme_to_id = data_artifacts["phoneme_to_id"]
    psv_shift_to_index = data_artifacts["psv_shift_to_index"]
    PAD_IDX = data_artifacts["PAD_IDX"]
    LANG_TOKENS = data_artifacts["LANG_TOKENS"]
    LANGUAGES = config["languages"]

    # --- 2. Load and Split Noisy Training Data ---
    # The df we received *is* the full noisy training data
    df_noisy_full = data_artifacts["df"]

    print("Shuffling and splitting noisy data...")
    df_noisy_shuffled = df_noisy_full.sample(frac=1, random_state=42).reset_index(
        drop=True
    )

    val_size = int(len(df_noisy_shuffled) * config["val_split_percent"])
    train_size = len(df_noisy_shuffled) - val_size

    df_noisy_train = df_noisy_shuffled.iloc[:train_size]
    df_noisy_val = df_noisy_shuffled.iloc[train_size:]

    # --- 3. Load Test Data ---
    print(
        f"Loading test set from {config['data_dir'] + "/" + config['test_data_file']}..."
    )
    try:
        df_test = pd.read_csv(config["data_dir"] + "/" + config["test_data_file"])
    except FileNotFoundError:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(
            f"ERROR: Test file not found at: {config['data_dir'] + "/" + config['test_data_file']}"
        )
        print("Please run verification and save your file to that location.")
        print("Creating an EMPTY test loader as a placeholder.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Create an empty df with the same columns to prevent crashes
        df_test = pd.DataFrame(columns=df_noisy_full.columns)

    print("Data split complete:")
    print(f"  Noisy Training Set:   {len(df_noisy_train)} rows")
    print(f"  Noisy Validation Set: {len(df_noisy_val)} rows")
    print(f"  Test Set:        {len(df_test)} rows")

    # --- 4. Create Datasets ---
    train_dataset = PhonemeDataset(
        df_noisy_train, phoneme_to_id, psv_shift_to_index, LANGUAGES, LANG_TOKENS
    )
    val_dataset = PhonemeDataset(
        df_noisy_val, phoneme_to_id, psv_shift_to_index, LANGUAGES, LANG_TOKENS
    )
    test_dataset = PhonemeDataset(
        df_test, phoneme_to_id, psv_shift_to_index, LANGUAGES, LANG_TOKENS
    )

    collate_with_padding = partial(collate_fn, pad_idx=PAD_IDX)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_with_padding,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_with_padding,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_with_padding,
    )
    print("DataLoaders created.")

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def build_model_and_tools(config, data_artifacts, device):
    """Initializes the model, loss functions, and optimizer."""

    print("Initializing model...")
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

    # Loss for translation
    translation_loss_fn = nn.CrossEntropyLoss(ignore_index=data_artifacts["PAD_IDX"])

    try:
        psv_weights = torch.load("data/processed/psv_weights.pt", map_location=device)
        print("Successfully loaded 'psv_weights.pt' for the loss function.")
    except FileNotFoundError:
        print("Warning: 'psv_weights.pt' not found. PSV loss will not be weighted by frequency.")
        psv_weights = None

    # Loss for rules (auxiliary head)
    pos_weight = torch.tensor([config["positive_weight"]], device=device)
    rules_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,  # This handles the 1s vs 0s imbalance
        weight=psv_weights      # This handles the "importance" of each rule
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    print(f"Optimizer initialized with weight decay: {config['weight_decay']}")

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    print("Learning rate scheduler (ReduceLROnPlateau) initialized.")

    return model, {
        "loss_fns": {"translation": translation_loss_fn, "rules": rules_loss_fn},
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


def train_one_epoch(
    model, loader, tools, config, data_artifacts, device, use_amp, scaler, epoch_num
):
    """Runs a single epoch of training."""
    model.train()
    train_loss_trans_epoch = 0.0
    train_loss_rules_epoch = 0.0

    PAD_IDX = data_artifacts["PAD_IDX"]
    PHONEME_VOCAB_SIZE = data_artifacts["PHONEME_VOCAB_SIZE"]

    optimizer = tools["optimizer"]
    translation_loss_fn = tools["loss_fns"]["translation"]
    rules_loss_fn = tools["loss_fns"]["rules"]

    for src, tgt, psm in tqdm(loader, desc=f"Epoch {epoch_num:03d} [Train]"):
        src, tgt, psm = src.to(device), tgt.to(device), psm.to(device)

        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]

        src_mask, tgt_pad_mask, causal_mask = create_masks(
            src, tgt_input, PAD_IDX, device
        )

        with autocast(device_type=device.type, enabled=use_amp):
            trans_logits, psm_logits = model(
                src,
                tgt_input,
                src_padding_mask=src_mask,
                tgt_padding_mask=tgt_pad_mask,
                tgt_mask=causal_mask,
            )

            loss_trans_logits = trans_logits.reshape(-1, PHONEME_VOCAB_SIZE)
            loss_trans_target = tgt_output.transpose(0, 1).reshape(-1)

            loss_trans = translation_loss_fn(loss_trans_logits, loss_trans_target)
            loss_rules = rules_loss_fn(psm_logits, psm)

            train_loss_trans_epoch += loss_trans.item()
            train_loss_rules_epoch += loss_rules.item()

            total_loss = loss_trans + (config["psv_loss_lambda"] * loss_rules)

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

    avg_loss_trans = train_loss_trans_epoch / len(loader)
    avg_loss_rules = train_loss_rules_epoch / len(loader)
    avg_total_loss = avg_loss_trans + (config["psv_loss_lambda"] * avg_loss_rules)

    return avg_total_loss, avg_loss_trans, avg_loss_rules


def validate_one_epoch(
    model, loader, tools, config, data_artifacts, device, use_amp, epoch_num
):
    """Runs a single epoch of validation."""
    model.eval()
    val_loss_trans_epoch = 0.0
    val_loss_rules_epoch = 0.0
    epoch_val_preds = []
    epoch_val_targets = []

    PAD_IDX = data_artifacts["PAD_IDX"]
    PHONEME_VOCAB_SIZE = data_artifacts["PHONEME_VOCAB_SIZE"]

    translation_loss_fn = tools["loss_fns"]["translation"]
    rules_loss_fn = tools["loss_fns"]["rules"]

    with torch.no_grad():
        for src, tgt, psm in tqdm(loader, desc=f"Epoch {epoch_num:03d} [Val]  "):
            src, tgt, psm = src.to(device), tgt.to(device), psm.to(device)

            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]

            src_mask, tgt_pad_mask, causal_mask = create_masks(
                src, tgt_input, PAD_IDX, device
            )

            with autocast(device_type=device.type, enabled=use_amp):
                trans_logits, psm_logits = model(
                    src,
                    tgt_input,
                    src_padding_mask=src_mask,
                    tgt_padding_mask=tgt_pad_mask,
                    tgt_mask=causal_mask,
                )

                loss_trans_logits = trans_logits.reshape(-1, PHONEME_VOCAB_SIZE)
                loss_trans_target = tgt_output.transpose(0, 1).reshape(-1)
                loss_trans = translation_loss_fn(loss_trans_logits, loss_trans_target)
                loss_rules = rules_loss_fn(psm_logits, psm)

            val_loss_trans_epoch += loss_trans.item()
            val_loss_rules_epoch += loss_rules.item()

            preds = torch.sigmoid(psm_logits)
            binary_preds = (preds > 0.5).cpu().numpy()
            epoch_val_preds.append(binary_preds)
            epoch_val_targets.append(psm.cpu().numpy().astype(int))

    avg_loss_trans = val_loss_trans_epoch / len(loader)
    avg_loss_rules = val_loss_rules_epoch / len(loader)
    avg_total_loss = avg_loss_trans + (config["psv_loss_lambda"] * avg_loss_rules)

    if epoch_val_preds:
        all_epoch_preds = np.concatenate(epoch_val_preds, axis=0)
        all_epoch_targets = np.concatenate(epoch_val_targets, axis=0)
        val_f1_micro = f1_score(
            all_epoch_targets, all_epoch_preds, average="micro", zero_division=0
        )
        val_f1_macro = f1_score(
            all_epoch_targets, all_epoch_preds, average="macro", zero_division=0
        )
    else:
        val_f1_micro = 0.0
        val_f1_macro = 0.0

    return avg_total_loss, avg_loss_trans, avg_loss_rules, val_f1_micro, val_f1_macro


def run_training_loop(config, model, tools, loaders, data_artifacts, device, use_amp):
    """The main outer loop for all epochs."""
    print("\n--- Starting Training ---")

    scheduler = tools["scheduler"]
    scaler = GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_trans_loss": [],
        "train_rules_loss": [],
        "val_trans_loss": [],
        "val_rules_loss": [],
        "val_f1_micro": [],
        "val_f1_macro": [],
    }

    for epoch in range(1, config["num_epochs"] + 1):

        # --- Training ---
        avg_train_loss, avg_train_trans, avg_train_rules = train_one_epoch(
            model,
            loaders["train"],
            tools,
            config,
            data_artifacts,
            device,
            use_amp,
            scaler,
            epoch,
        )

        # --- Validation ---
        avg_val_loss, avg_val_trans, avg_val_rules, f1_micro, f1_macro = (
            validate_one_epoch(
                model,
                loaders["val"],
                tools,
                config,
                data_artifacts,
                device,
                use_amp,
                epoch,
            )
        )

        scheduler.step(avg_val_loss)

        # --- Log History ---
        history["train_loss"].append(avg_train_loss)
        history["train_trans_loss"].append(avg_train_trans)
        history["train_rules_loss"].append(avg_train_rules)
        history["val_loss"].append(avg_val_loss)
        history["val_trans_loss"].append(avg_val_trans)
        history["val_rules_loss"].append(avg_val_rules)
        history["val_f1_micro"].append(f1_micro)
        history["val_f1_macro"].append(f1_macro)

        # --- Print Epoch Results ---
        print(
            f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} (T:{avg_train_trans:.4f} R:{avg_train_rules:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} (T:{avg_val_trans:.4f} R:{avg_val_rules:.4f}) | "
            f"Val F1 Micro: {f1_micro:.4f} | Val F1 Macro: {f1_macro:.4f}"
        )

        # --- Early Stopping & Model Saving ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("  -> Validation loss improved. Saving full checkpoint...")
            save_data = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "data_artifacts": data_artifacts,
            }
            torch.save(save_data, config["model_save_path"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(
                f"  -> Validation loss did not improve for {epochs_without_improvement} epoch(s)."
            )
            if epochs_without_improvement >= config["early_stopping_patience"]:
                print(f"\n--- Early stopping triggered after {epoch} epochs ---")
                break

    print("--- Training Complete ---")
    return history


def plot_training_history(history, save_path):
    """Saves a plot of all training and validation loss components."""

    print("\nPlotting loss history...")
    actual_epochs_run = len(history["train_loss"])
    if actual_epochs_run == 0:
        print("No history to plot. Skipping plot generation.")
        return

    epochs_range = range(1, actual_epochs_run + 1)

    plt.figure(figsize=(12, 8))

    # Plot Total Losses
    plt.subplot(2, 1, 1)
    plt.plot(
        epochs_range, history["train_loss"], label="Training Loss (Total Weighted)"
    )
    plt.plot(
        epochs_range, history["val_loss"], label="Validation Loss (Total Weighted)"
    )
    plt.title("Total Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot Loss Components
    plt.subplot(2, 1, 2)
    plt.plot(
        epochs_range,
        history["train_trans_loss"],
        label="Train Translation Loss",
        linestyle="--",
    )
    plt.plot(
        epochs_range,
        history["train_rules_loss"],
        label="Train Rules Loss",
        linestyle="--",
    )
    plt.plot(
        epochs_range,
        history["val_trans_loss"],
        label="Val Translation Loss",
        linestyle=":",
    )
    plt.plot(
        epochs_range, history["val_rules_loss"], label="Val Rules Loss", linestyle=":"
    )
    plt.title("Loss Components")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Loss history plot saved to {save_path}")


# --- 3. Main "Manager" Function ---


def main():

    # --- 1. Load Config & Setup ---
    # CONFIG is global, so we can access it
    device, use_amp = setup_device_and_amp()

    # --- 2. Load Data ---
    data_artifacts = load_data_and_vocabs(CONFIG)
    if data_artifacts is None:
        return  # Exit if data loading failed

    # --- 3. Create DataLoaders ---
    loaders = create_dataloaders(CONFIG, data_artifacts)

    # --- 4. Build Model & Tools ---
    model, tools = build_model_and_tools(CONFIG, data_artifacts, device)

    # --- 5. Run Training ---
    history = run_training_loop(
        CONFIG, model, tools, loaders, data_artifacts, device, use_amp
    )

    # --- 6. Plot Results ---
    plot_training_history(history, CONFIG["plot_save_path"])

    print("\n--- Process Finished ---")


# --- This makes the script runnable from the command line ---
if __name__ == "__main__":
    main()
