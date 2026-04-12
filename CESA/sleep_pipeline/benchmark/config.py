"""Reproducibility configuration for benchmark experiments.

Centralises random seed management, environment logging, and
experiment configuration in a serialisable dataclass.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_YAML_AVAILABLE = False
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    pass


@dataclass
class BenchmarkConfig:
    """Full experiment configuration."""

    dataset: str = "sleep_edf"
    data_path: str = ""
    n_subjects: int = -1
    n_folds: int = 5
    epoch_duration_s: float = 30.0
    target_sfreq: float = 100.0
    backends: List[str] = field(default_factory=lambda: [
        "aasm_rules", "ml", "ml_hmm",
    ])
    ml_model_type: str = "hgb"
    random_seed: int = 42
    output_dir: str = "results"
    figure_dpi: int = 300
    figure_format: str = "pdf"
    bootstrap_n: int = 1000
    bootstrap_ci: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        if p.suffix in (".yaml", ".yml") and _YAML_AVAILABLE:
            p.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
        else:
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str) -> "BenchmarkConfig":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml") and _YAML_AVAILABLE:
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def config_hash(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


def fix_all_seeds(seed: int) -> None:
    """Fix random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def log_environment() -> Dict[str, Any]:
    """Capture environment metadata for reproducibility."""
    env: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }

    # Git hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            env["git_hash"] = result.stdout.strip()
    except Exception:
        pass

    # Key package versions
    for pkg in ("numpy", "scipy", "sklearn", "mne", "torch", "matplotlib", "pandas"):
        try:
            mod = __import__(pkg)
            env[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env[f"{pkg}_version"] = "not_installed"

    # scikit-learn special case
    try:
        import sklearn
        env["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass

    return env


def setup_output_dir(config: BenchmarkConfig) -> Path:
    """Create timestamped output directory and save config + environment."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(config.output_dir) / f"{config.dataset}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    config.save(str(out / "config.yaml" if _YAML_AVAILABLE else out / "config.json"))

    env = log_environment()
    (out / "environment.json").write_text(
        json.dumps(env, indent=2), encoding="utf-8",
    )

    return out
