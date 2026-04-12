"""Hidden Markov Model for sequential sleep-stage decoding.

Enforces biologically plausible stage transitions via a constrained
transition matrix and decodes the optimal state sequence using the
Viterbi algorithm (log-space for numerical stability).

The emission model consumes class-probability vectors produced by the ML
scorer (or confidence vectors from rule-based scoring) rather than raw
features, making this module model-agnostic.

Design rationale
----------------
* **Transition matrix** is initialised from AASM-compatible priors:
  high self-transition (~0.90), forbidden jumps (W->N3, R->N3) set to
  near-zero.  Values are derived from healthy-adult hypnogram statistics
  (Berthomier et al. 2007, Liang et al. 2012).
* **Viterbi decoding** in log-space avoids underflow on long recordings.
* Optional **Baum-Welch** refinement adapts the matrix to a local cohort.

References
----------
Berthomier, C. et al. (2007). Automatic analysis of single-channel
sleep EEG.  IEEE Trans Biomed Eng.
Rabiner, L. R. (1989). A tutorial on hidden Markov models.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from .contracts import Epoch, ScoringResult, StageLabel

logger = logging.getLogger(__name__)

# State ordering: W=0, N1=1, N2=2, N3=3, R=4  (matches contracts)
_N_STATES = 5
_STATE_NAMES = ["W", "N1", "N2", "N3", "R"]
_STATE_TO_IDX = {s: i for i, s in enumerate(_STATE_NAMES)}


# ---------------------------------------------------------------------------
# Biologically constrained transition matrix
# ---------------------------------------------------------------------------

def build_aasm_transition_matrix() -> np.ndarray:
    """Return a 5x5 AASM-plausible transition matrix.

    Rows = current state, columns = next state.  Each row sums to 1.

    Constraints enforced:
    - W  -> N3 forbidden  (must transit through N1/N2)
    - R  -> N3 forbidden
    - N3 -> R  forbidden  (must transit through N2/N1)
    - N3 -> N1 very rare
    - High self-transition probability for stability
    """
    # fmt: off
    A = np.array([
        #   W      N1     N2     N3     R
        [0.900, 0.060, 0.020, 0.000, 0.020],  # W  -> mostly stays W
        [0.050, 0.750, 0.150, 0.010, 0.040],  # N1 -> often -> N2, sometimes -> W/R
        [0.020, 0.030, 0.850, 0.080, 0.020],  # N2 -> stable, can deepen to N3 or lighten
        [0.010, 0.005, 0.085, 0.900, 0.000],  # N3 -> very stable, exits via N2
        [0.040, 0.030, 0.020, 0.000, 0.910],  # R  -> stable, exits via W/N1/N2
    ], dtype=np.float64)
    # fmt: on

    # Safety: normalise rows
    A = A / A.sum(axis=1, keepdims=True)
    return A


def build_initial_probs() -> np.ndarray:
    """Initial state distribution (recording typically starts in Wake)."""
    pi = np.array([0.70, 0.15, 0.10, 0.02, 0.03], dtype=np.float64)
    return pi / pi.sum()


# ---------------------------------------------------------------------------
# SleepHMM
# ---------------------------------------------------------------------------

class SleepHMM:
    """5-state HMM with Viterbi decoding for sleep staging.

    Parameters
    ----------
    transition_matrix : (5, 5) array, optional
        Row-stochastic transition matrix.  Defaults to AASM priors.
    initial_probs : (5,) array, optional
        Initial state distribution.
    """

    def __init__(
        self,
        transition_matrix: Optional[np.ndarray] = None,
        initial_probs: Optional[np.ndarray] = None,
    ) -> None:
        self.A = transition_matrix if transition_matrix is not None else build_aasm_transition_matrix()
        self.pi = initial_probs if initial_probs is not None else build_initial_probs()
        assert self.A.shape == (_N_STATES, _N_STATES)
        assert self.pi.shape == (_N_STATES,)
        self._log_A = np.log(np.clip(self.A, 1e-12, None))
        self._log_pi = np.log(np.clip(self.pi, 1e-12, None))

    # -----------------------------------------------------------------------
    # Viterbi decoding
    # -----------------------------------------------------------------------

    def decode_viterbi(self, emission_probs: np.ndarray) -> np.ndarray:
        """Find the most likely state sequence given emission probabilities.

        Parameters
        ----------
        emission_probs : (T, 5) array
            Per-epoch class probabilities (e.g. from ``model.predict_proba``).
            Each row should sum to ~1.

        Returns
        -------
        (T,) int array
            Optimal state indices (0=W, 1=N1, 2=N2, 3=N3, 4=R).
        """
        T, K = emission_probs.shape
        assert K == _N_STATES, f"Expected {_N_STATES} states, got {K}"

        log_emission = np.log(np.clip(emission_probs, 1e-12, None))

        # Viterbi tables
        V = np.zeros((T, K), dtype=np.float64)  # log-probability
        backptr = np.zeros((T, K), dtype=int)

        # Initialisation
        V[0] = self._log_pi + log_emission[0]

        # Forward pass
        for t in range(1, T):
            for j in range(K):
                candidates = V[t - 1] + self._log_A[:, j]
                backptr[t, j] = int(np.argmax(candidates))
                V[t, j] = candidates[backptr[t, j]] + log_emission[t, j]

        # Backtrack
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(V[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = backptr[t + 1, path[t + 1]]

        return path

    def decode_labels(self, emission_probs: np.ndarray) -> List[StageLabel]:
        """Decode and return StageLabel list."""
        path = self.decode_viterbi(emission_probs)
        return [StageLabel.from_string(_STATE_NAMES[s]) for s in path]

    # -----------------------------------------------------------------------
    # Baum-Welch fitting (EM)
    # -----------------------------------------------------------------------

    def fit(
        self,
        sequences: List[np.ndarray],
        *,
        max_iter: int = 50,
        tol: float = 1e-4,
        min_transition: float = 1e-6,
    ) -> "SleepHMM":
        """Refine transition matrix from labelled emission-probability sequences.

        Parameters
        ----------
        sequences : list of (T_i, 5) arrays
            Multiple recordings' emission probabilities.
        max_iter : int
            Maximum EM iterations.
        tol : float
            Convergence threshold on log-likelihood change.
        min_transition : float
            Floor for transition probabilities (preserves biological constraints).

        Returns
        -------
        self (fitted)
        """
        prev_ll = -np.inf

        for it in range(max_iter):
            A_num = np.zeros((_N_STATES, _N_STATES), dtype=np.float64)
            A_den = np.zeros(_N_STATES, dtype=np.float64)
            pi_num = np.zeros(_N_STATES, dtype=np.float64)
            total_ll = 0.0

            for emission_probs in sequences:
                T = len(emission_probs)
                if T < 2:
                    continue
                log_emission = np.log(np.clip(emission_probs, 1e-12, None))

                # Forward
                alpha = np.zeros((T, _N_STATES))
                alpha[0] = self._log_pi + log_emission[0]
                for t in range(1, T):
                    for j in range(_N_STATES):
                        alpha[t, j] = _logsumexp(alpha[t - 1] + self._log_A[:, j]) + log_emission[t, j]

                # Backward
                beta = np.zeros((T, _N_STATES))
                for t in range(T - 2, -1, -1):
                    for i in range(_N_STATES):
                        beta[t, i] = _logsumexp(
                            self._log_A[i, :] + log_emission[t + 1] + beta[t + 1]
                        )

                # Log-likelihood
                ll = _logsumexp(alpha[-1])
                total_ll += ll

                # Accumulate expected transitions
                for t in range(T - 1):
                    for i in range(_N_STATES):
                        for j in range(_N_STATES):
                            xi = alpha[t, i] + self._log_A[i, j] + log_emission[t + 1, j] + beta[t + 1, j] - ll
                            A_num[i, j] += np.exp(xi)
                        A_den[i] += np.exp(_logsumexp(
                            alpha[t, i] + beta[t, i] - ll + np.zeros(_N_STATES)
                        ))

                # Initial state
                gamma_0 = alpha[0] + beta[0] - ll
                pi_num += np.exp(gamma_0)

            # Update parameters
            for i in range(_N_STATES):
                if A_den[i] > 0:
                    self.A[i] = np.clip(A_num[i] / A_den[i], min_transition, None)
                self.A[i] /= self.A[i].sum()

            pi_sum = pi_num.sum()
            if pi_sum > 0:
                self.pi = pi_num / pi_sum

            self._log_A = np.log(np.clip(self.A, 1e-12, None))
            self._log_pi = np.log(np.clip(self.pi, 1e-12, None))

            logger.debug("Baum-Welch iter %d: LL=%.4f", it, total_ll)
            if abs(total_ll - prev_ll) < tol:
                logger.info("Baum-Welch converged at iteration %d", it)
                break
            prev_ll = total_ll

        return self


# ---------------------------------------------------------------------------
# Public pipeline function
# ---------------------------------------------------------------------------

def hmm_decode_scoring(
    result: ScoringResult,
    *,
    hmm: Optional[SleepHMM] = None,
) -> ScoringResult:
    """Apply HMM Viterbi decoding to refine a ScoringResult.

    Takes the confidence/probability vectors from each epoch and re-decodes
    the entire sequence with the HMM's transition constraints.

    Parameters
    ----------
    result : ScoringResult
        Initial scoring (from ML or rule-based backend).
    hmm : SleepHMM, optional
        Pre-configured HMM.  Defaults to AASM-prior HMM.

    Returns
    -------
    ScoringResult with HMM-decoded stages.
    """
    if hmm is None:
        hmm = SleepHMM()

    if not result.epochs:
        return result

    # Build emission matrix from epoch confidences
    emission = _build_emission_matrix(result)
    decoded = hmm.decode_labels(emission)

    new_epochs = []
    for ep, new_stage in zip(result.epochs, decoded):
        new_ep = Epoch(
            index=ep.index,
            start_s=ep.start_s,
            duration_s=ep.duration_s,
            features=ep.features,
            stage=new_stage,
            confidence=ep.confidence,
            decision_reason=ep.decision_reason,
        )
        if new_stage != ep.stage:
            new_ep.decision_reason += f"|hmm_from_{ep.stage.value}"
        new_epochs.append(new_ep)

    return ScoringResult(
        epochs=new_epochs,
        events=result.events,
        epoch_duration_s=result.epoch_duration_s,
        backend=result.backend + "+hmm",
        metadata={**result.metadata, "hmm_applied": True},
    )


def _build_emission_matrix(result: ScoringResult) -> np.ndarray:
    """Build (T, 5) emission probability matrix from a ScoringResult.

    If the epoch has a single hard prediction with confidence, we build
    a soft distribution: confidence on the predicted class, remainder
    spread uniformly.
    """
    T = len(result.epochs)
    emission = np.full((T, _N_STATES), 1e-3, dtype=np.float64)

    for i, ep in enumerate(result.epochs):
        idx = _STATE_TO_IDX.get(ep.stage.value)
        if idx is None:
            continue
        conf = max(ep.confidence, 0.2)  # floor to avoid over-confident hard labels
        emission[i, idx] = conf
        remainder = (1.0 - conf) / max(_N_STATES - 1, 1)
        for j in range(_N_STATES):
            if j != idx:
                emission[i, j] = remainder

    # Normalise rows
    emission = emission / emission.sum(axis=1, keepdims=True)
    return emission


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    mx = np.max(x)
    if np.isinf(mx):
        return float(mx)
    return float(mx + np.log(np.sum(np.exp(x - mx))))
