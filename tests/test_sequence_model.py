"""Tests for CESA.sleep_pipeline.sequence_model (HMM)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CESA.sleep_pipeline.contracts import Epoch, ScoringResult, StageLabel
from CESA.sleep_pipeline.sequence_model import (
    SleepHMM,
    _build_emission_matrix,
    _logsumexp,
    build_aasm_transition_matrix,
    build_initial_probs,
    hmm_decode_scoring,
)


class TestTransitionMatrix:
    def test_rows_sum_to_one(self):
        A = build_aasm_transition_matrix()
        np.testing.assert_allclose(A.sum(axis=1), 1.0, atol=1e-10)

    def test_shape(self):
        A = build_aasm_transition_matrix()
        assert A.shape == (5, 5)

    def test_forbidden_transitions_near_zero(self):
        A = build_aasm_transition_matrix()
        # W->N3 (row 0, col 3) should be ~0
        assert A[0, 3] < 0.01
        # R->N3 (row 4, col 3) should be ~0
        assert A[4, 3] < 0.01
        # N3->R (row 3, col 4) should be ~0
        assert A[3, 4] < 0.01

    def test_self_transitions_dominant(self):
        A = build_aasm_transition_matrix()
        for i in range(5):
            assert A[i, i] >= 0.7


class TestInitialProbs:
    def test_sums_to_one(self):
        pi = build_initial_probs()
        np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-10)

    def test_wake_dominant(self):
        pi = build_initial_probs()
        assert pi[0] > 0.5  # Wake should be most likely at start


class TestSleepHMM:
    def test_viterbi_pure_wake(self):
        hmm = SleepHMM()
        # 10 epochs with high probability of Wake
        emission = np.tile([0.9, 0.02, 0.02, 0.02, 0.04], (10, 1))
        path = hmm.decode_viterbi(emission)
        assert np.all(path == 0)  # All Wake

    def test_viterbi_pure_n3(self):
        hmm = SleepHMM()
        # After a Wake->N1->N2 ramp, strong N3 emission
        emission = np.zeros((20, 5))
        emission[0:2, 0] = 0.9  # Wake
        emission[0:2, 1:] = 0.025
        emission[2:4, 1] = 0.8  # N1
        emission[2:4, [0, 2, 3, 4]] = 0.05
        emission[4:6, 2] = 0.8  # N2
        emission[4:6, [0, 1, 3, 4]] = 0.05
        emission[6:, 3] = 0.9  # N3
        emission[6:, [0, 1, 2, 4]] = 0.025

        path = hmm.decode_viterbi(emission)
        # Last epochs should be N3
        assert path[-1] == 3

    def test_viterbi_prevents_impossible_jump(self):
        hmm = SleepHMM()
        # Longer sequence: clear Wake start, then gradual shift to N3
        # With enough Wake epochs at start, the HMM commits to Wake
        emission = np.zeros((8, 5))
        emission[0] = [0.9, 0.02, 0.02, 0.02, 0.04]  # Clear Wake
        emission[1] = [0.9, 0.02, 0.02, 0.02, 0.04]  # Clear Wake
        emission[2] = [0.9, 0.02, 0.02, 0.02, 0.04]  # Clear Wake
        emission[3] = [0.05, 0.05, 0.05, 0.80, 0.05]  # Sudden N3
        emission[4] = [0.05, 0.05, 0.05, 0.80, 0.05]  # Strong N3
        emission[5] = [0.05, 0.05, 0.05, 0.80, 0.05]  # Strong N3
        emission[6] = [0.05, 0.05, 0.05, 0.80, 0.05]  # Strong N3
        emission[7] = [0.05, 0.05, 0.05, 0.80, 0.05]  # Strong N3

        path = hmm.decode_viterbi(emission)
        assert path[0] == 0  # First is Wake
        assert path[1] == 0  # Second is Wake
        # At the transition point (epoch 3), the HMM should NOT jump
        # directly to N3 (which requires going through N1->N2)
        if path[2] == 0 and path[3] == 3:
            pytest.fail("HMM allowed W->N3 direct transition")

    def test_decode_labels(self):
        hmm = SleepHMM()
        emission = np.tile([0.8, 0.05, 0.05, 0.05, 0.05], (3, 1))
        labels = hmm.decode_labels(emission)
        assert all(isinstance(l, StageLabel) for l in labels)
        assert all(l == StageLabel.W for l in labels)


class TestHmmDecodeScoringResult:
    def _make_result(self):
        stages = [StageLabel.W, StageLabel.W, StageLabel.N1, StageLabel.N2, StageLabel.N3]
        epochs = [
            Epoch(index=i, start_s=i * 30.0, stage=s, confidence=0.8,
                  decision_reason=f"test_{s.value}")
            for i, s in enumerate(stages)
        ]
        return ScoringResult(epochs=epochs, epoch_duration_s=30.0, backend="test")

    def test_basic_decode(self):
        result = self._make_result()
        decoded = hmm_decode_scoring(result)
        assert len(decoded.epochs) == 5
        assert "hmm" in decoded.backend

    def test_preserves_epoch_count(self):
        result = self._make_result()
        decoded = hmm_decode_scoring(result)
        assert len(decoded.epochs) == len(result.epochs)

    def test_empty_result(self):
        result = ScoringResult()
        decoded = hmm_decode_scoring(result)
        assert len(decoded.epochs) == 0


class TestBuildEmission:
    def test_shape(self):
        epochs = [
            Epoch(index=i, start_s=i * 30.0, stage=StageLabel.W, confidence=0.8)
            for i in range(5)
        ]
        result = ScoringResult(epochs=epochs)
        em = _build_emission_matrix(result)
        assert em.shape == (5, 5)

    def test_rows_sum_to_one(self):
        epochs = [
            Epoch(index=i, start_s=i * 30.0, stage=StageLabel.N2, confidence=0.7)
            for i in range(3)
        ]
        result = ScoringResult(epochs=epochs)
        em = _build_emission_matrix(result)
        np.testing.assert_allclose(em.sum(axis=1), 1.0, atol=1e-10)


class TestLogsumexp:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        result = _logsumexp(x)
        expected = np.log(np.sum(np.exp(x)))
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_large_values(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        result = _logsumexp(x)
        assert np.isfinite(result)
