"""Tests for the event-locked coupling figures (average, PETH, TFR)."""

from __future__ import annotations

import numpy as np
import pytest

from CESA import coupling_event_plots as ep
from CESA.sleep_pipeline.so_spindle_coupling import detect_so_spindle_coupling
from tests.test_so_spindle_coupling import SFREQ, synthetic_recording


@pytest.fixture
def coupled_signal():
    signal, _ = synthetic_recording(seed=1)
    result = detect_so_spindle_coupling(signal[None, :], SFREQ, ["Cz"])
    troughs, peaks = ep.event_indices_from_dicts(result.events, SFREQ, "Cz")
    return signal, troughs, peaks


def test_spindle_locked_average_has_axes(coupled_signal):
    signal, _troughs, peaks = coupled_signal
    fig = ep.spindle_locked_average(signal, SFREQ, peaks)
    assert fig.axes
    assert fig.axes[0].lines


def test_peth_counts_near_upstate(coupled_signal):
    _signal, troughs, peaks = coupled_signal
    fig = ep.peri_event_histogram(troughs, peaks, SFREQ)
    assert fig.axes
    # Injected bursts sit ~0.5 s after the trough (positive peak of a 1 s SO).
    patches = fig.axes[0].patches
    assert patches


def test_tfr_returns_image(coupled_signal):
    signal, troughs, _peaks = coupled_signal
    fig = ep.so_trough_tfr(signal, SFREQ, troughs, min_events=8)
    assert fig.axes
    assert fig.axes[0].collections or fig.axes[0].images


def test_too_few_events_draw_a_warning_not_a_crash():
    fig = ep.spindle_locked_average(np.zeros(1000), SFREQ, [100, 200], min_events=8)
    assert fig.axes
    fig2 = ep.so_trough_tfr(np.zeros(5000), SFREQ, [500], min_events=8)
    assert fig2.axes


def test_event_indices_ignore_other_channels_and_yasa(coupled_signal):
    signal, troughs, peaks = coupled_signal
    result = detect_so_spindle_coupling(signal[None, :], SFREQ, ["Cz"])
    from CESA.sleep_pipeline.contracts import EventType, ScoredEvent

    mixed = list(result.events) + [
        ScoredEvent(
            EventType.SLOW_WAVE, 1.0, 1.0,
            metadata={"channel": "C4", "detector": "cesa_classic", "trough_s": 1.0},
        ),
        ScoredEvent(
            EventType.SPINDLE, 2.0, 0.8,
            metadata={"channel": "Cz", "detector": "yasa", "peak_s": 2.0},
        ),
    ]
    got_troughs, got_peaks = ep.event_indices_from_dicts(mixed, SFREQ, "Cz")
    assert got_troughs == troughs
    assert got_peaks == peaks


def test_cohort_accumulator_writes_only_requested(tmp_path, coupled_signal):
    signal, troughs, peaks = coupled_signal
    acc = ep.CohortEventPlots()
    acc.add_recording(
        "Cz", signal, SFREQ, troughs, peaks,
        want_tfr=False, want_averages=True,
    )
    written = acc.save(tmp_path, tfr=False, averages=True)
    names = {p.split("\\")[-1].split("/")[-1] for p in written}
    assert "coupling_moyenne.png" in names
    assert "coupling_peth.png" in names
    assert "coupling_tfr.png" not in names
