"""Tests for core.telemetry mode checkpoints."""

from __future__ import annotations

import pytest

from core.telemetry import TelemetryRecorder


def test_final_validation_lazy_only_satisfies_or_semantics() -> None:
    rec = TelemetryRecorder()
    rec.expect_modes(["lazy", "precomputed"])
    rec.mark_mode("lazy")
    rec.flush(final=True)


def test_final_validation_precomputed_only_satisfies_or_semantics() -> None:
    rec = TelemetryRecorder()
    rec.expect_modes(["lazy", "precomputed"])
    rec.mark_mode("precomputed")
    rec.flush(final=True)


def test_final_validation_raises_when_no_expected_mode_observed() -> None:
    rec = TelemetryRecorder()
    rec.expect_modes(["lazy", "precomputed"])
    try:
        with pytest.raises(RuntimeError, match="none of the expected modes"):
            rec.flush(final=True)
    finally:
        # Chaque instance enregistre atexit(flush) ; vider les attentes pour l’arrêt propre.
        rec.expect_modes(())


def test_mark_mode_rejects_observation_outside_expected_set() -> None:
    rec = TelemetryRecorder()
    rec.expect_modes(["lazy", "precomputed"])
    try:
        with pytest.raises(RuntimeError, match="observed"):
            rec.mark_mode("raw")
    finally:
        rec.expect_modes(())
