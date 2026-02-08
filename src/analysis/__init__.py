"""Analysis and visualization scripts for interpreting trained models.

These scripts generally load a checkpoint from `src/training/train.py` and use
the saved vocabularies/mappings in `data_artifacts` to:
- aggregate per-language PSV/PSM predictions,
- compare predictions to ground truth,
- visualize attention, and
- produce PCA/drift plots and clustering figures.
"""
