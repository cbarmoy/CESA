"""Deep-learning sleep scoring skeleton.

**Status: experimental** -- requires training on a local cohort before
clinical use.

Modules
-------
dataset   PyTorch Dataset for multi-channel 30-s epochs.
model     CNN/LSTM baseline architecture.
train     Training loop with validation and early stopping.
infer     Inference entrypoint returning a CESA ScoringResult.
"""
