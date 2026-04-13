"""
CESA Sleep Pipeline -- AASM-aligned modular sleep scoring.

Sub-modules
-----------
contracts      Canonical types (Epoch, StageLabel, ScoringResult, ...)
preprocessing  Channel normalisation, resampling, 30-s epoching
features       Time/frequency/statistical feature extraction per epoch
rules_aasm     Explicit rule-based stage classification (AASM criteria)
events         Arousal / apnea / hypopnea detection
evaluation     Accuracy, Cohen's kappa, confusion matrix, per-stage metrics
transition     Bridge between legacy CESA scoring and this pipeline
"""

__version__ = "0.0beta1.1"
