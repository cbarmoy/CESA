"""The ML backend must feed a model the columns it was trained on.

Feature dicts grow over time (a new descriptor, a modality-availability flag).
Since the matrix columns are the sorted feature names, an added key shifts every
column: without alignment a previously trained model keeps predicting, silently,
on scrambled inputs.
"""

from __future__ import annotations

import numpy as np

from CESA.sleep_pipeline.ml_scorer import _features_to_matrix, score_ml


def _feats(n=3, **extra):
    out = []
    for i in range(n):
        f = {"alpha": float(i), "beta": float(i) * 2, "delta": float(i) * 3}
        f.update(extra)
        out.append(f)
    return out


class _SpyModel:
    """Records the matrix it was asked to predict on."""

    def __init__(self, names):
        self._sleep_feature_names = list(names)
        self.seen = None

    def predict(self, X):
        self.seen = np.asarray(X)
        return np.zeros(self.seen.shape[0], dtype=int)


class TestColumnAlignment:
    def test_requested_names_fix_the_column_order(self):
        X, names = _features_to_matrix(_feats(), ["delta", "alpha"])
        assert names == ["delta", "alpha"]
        assert X[1].tolist() == [3.0, 1.0]

    def test_unknown_requested_name_becomes_zero(self):
        X, _ = _features_to_matrix(_feats(), ["alpha", "absent"])
        assert X[:, 1].tolist() == [0.0, 0.0, 0.0]

    def test_extra_feature_keys_are_dropped_not_inserted(self):
        trained = ["alpha", "beta", "delta"]
        X, names = _features_to_matrix(_feats(has_emg=0.0), trained)
        assert names == trained
        assert X.shape[1] == 3

    def test_score_ml_aligns_on_the_trained_names(self):
        model = _SpyModel(["alpha", "beta", "delta"])
        score_ml(
            _feats(has_emg=0.0, has_eog=1.0),
            model=model,
            use_temporal_context=False,
            apply_smoothing=False,
        )
        # Three trained columns, not the five now present in the features.
        assert model.seen.shape[1] == 3
        assert model.seen[1].tolist() == [1.0, 2.0, 3.0]

    def test_a_model_without_recorded_names_still_works(self):
        model = _SpyModel([])
        del model._sleep_feature_names
        score_ml(
            _feats(), model=model, use_temporal_context=False, apply_smoothing=False,
        )
        assert model.seen.shape[1] == 3
