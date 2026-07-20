"""Tests for EOG→EEG ocular cleaning (in-memory, non-destructive)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from CESA.ocular_cleaning import (
    bandpower,
    clean_channels,
    ocular_bandpower_ratio,
    regress_eog_from_eeg,
    slice_signal,
)


def _synth_contaminated(sfreq: float = 200.0, duration_s: float = 20.0, gain: float = 3.0):
    n = int(sfreq * duration_s)
    t = np.arange(n) / sfreq
    rng = np.random.default_rng(0)
    # Slow ocular-like artifact
    eog = 80.0 * np.sin(2 * np.pi * 0.8 * t) + 40.0 * np.sin(2 * np.pi * 1.5 * t)
    # Clean EEG: alpha + noise
    eeg_clean = 10.0 * np.sin(2 * np.pi * 10.0 * t) + rng.normal(0, 2.0, size=n)
    eeg_dirty = eeg_clean + gain * eog
    return eeg_dirty, eog, eeg_clean, sfreq


def test_regress_reduces_eog_correlation():
    eeg_dirty, eog, _eeg_clean, sfreq = _synth_contaminated()
    cleaned = regress_eog_from_eeg(eeg_dirty, [eog], sfreq)
    corr_before = abs(np.corrcoef(eeg_dirty, eog)[0, 1])
    corr_after = abs(np.corrcoef(cleaned, eog)[0, 1])
    assert corr_before > 0.7
    assert corr_after < corr_before * 0.3


def test_ocular_bandpower_decreases():
    eeg_dirty, eog, _eeg_clean, sfreq = _synth_contaminated()
    cleaned = regress_eog_from_eeg(eeg_dirty, [eog], sfreq)
    ratio = ocular_bandpower_ratio(eeg_dirty, cleaned, sfreq, band=(0.3, 4.0))
    assert ratio < -0.3  # at least 30% reduction in ocular band


def test_clean_channels_does_not_return_eog():
    eeg_dirty, eog, _, sfreq = _synth_contaminated()
    signals = {
        "C3-M2": (eeg_dirty, sfreq),
        "EOG gauche": (eog, sfreq),
        "EMG": (np.zeros_like(eog), sfreq),
    }
    out = clean_channels(signals, ["C3-M2"], ["EOG gauche"], sfreq=sfreq)
    assert set(out.keys()) == {"C3-M2"}
    assert "EOG gauche" not in out
    assert "EMG" not in out


def test_clean_channels_empty_without_eog():
    eeg_dirty, eog, _, sfreq = _synth_contaminated()
    signals = {"C3-M2": (eeg_dirty, sfreq)}
    assert clean_channels(signals, ["C3-M2"], ["EOG"], sfreq=sfreq) == {}


def test_multi_eog_refs():
    eeg_dirty, eog, _, sfreq = _synth_contaminated()
    eog2 = eog * 0.5 + 5.0 * np.sin(2 * np.pi * 0.5 * np.arange(len(eog)) / sfreq)
    cleaned = regress_eog_from_eeg(eeg_dirty, [eog, eog2], sfreq)
    assert cleaned.shape == eeg_dirty.shape
    assert bandpower(cleaned, sfreq, (0.3, 4.0)) < bandpower(eeg_dirty, sfreq, (0.3, 4.0))


def test_slice_signal():
    x = np.arange(1000, dtype=float)
    seg = slice_signal(x, sfreq=100.0, t0=2.0, duration=1.0)
    assert seg.shape == (100,)
    assert seg[0] == 200.0


def test_no_file_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Ocular cleaning must not create files on disk."""
    monkeypatch.chdir(tmp_path)
    eeg_dirty, eog, _, sfreq = _synth_contaminated()
    signals = {"C3": (eeg_dirty, sfreq), "EOG": (eog, sfreq)}
    cleaned = clean_channels(signals, ["C3"], ["EOG"], sfreq=sfreq)
    _ = ocular_bandpower_ratio(eeg_dirty, cleaned["C3"], sfreq)
    assert list(tmp_path.iterdir()) == []


def test_controller_apply_restore_memory_only():
    from CESA.app_controller import AppController

    eeg_dirty, eog, _, sfreq = _synth_contaminated()
    ctrl = AppController()
    ctrl.sfreq = sfreq
    ctrl.derivations = {
        "C3-M2": eeg_dirty.copy(),
        "EOG": eog.copy(),
    }
    cleaned = clean_channels(
        {"C3-M2": (eeg_dirty, sfreq), "EOG": (eog, sfreq)},
        ["C3-M2"],
        ["EOG"],
        sfreq=sfreq,
    )
    assert not ctrl.has_ocular_cleaning()
    n = ctrl.apply_ocular_cleaning(cleaned)
    assert n == 1
    assert ctrl.has_ocular_cleaning()
    assert not np.allclose(ctrl.derivations["C3-M2"], eeg_dirty)
    restored = ctrl.restore_ocular_backup()
    assert "C3-M2" in restored
    assert np.allclose(ctrl.derivations["C3-M2"], eeg_dirty)
    assert not ctrl.has_ocular_cleaning()
