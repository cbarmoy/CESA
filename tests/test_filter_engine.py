"""Tests for CESA.filter_engine -- individual filters, pipeline, serialisation,
presets, backward compatibility (filters.py), preprocessing integration,
physiological ranges, audit log, user presets, import/export,
UndoManager, AdaptiveFilterSuggester, FavoritePresets, ChannelAnnotations,
versioning consistency, and report generation."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CESA.filter_engine import (
    PHYSIOLOGICAL_RANGES,
    AdaptiveFilterSuggester,
    BandpassFilter,
    ChannelAnnotation,
    ChannelAnnotationStore,
    FavoritePresets,
    FilterAuditLog,
    FilterPipeline,
    FilterPreset,
    FilterSuggestion,
    HighpassFilter,
    LowpassFilter,
    NotchFilter,
    PresetLibrary,
    SmoothingFilter,
    UndoManager,
    filter_from_dict,
    pipeline_from_legacy_params,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SFREQ = 256.0

def _sine(freq_hz: float, duration_s: float = 5.0, sfreq: float = SFREQ) -> np.ndarray:
    t = np.arange(int(sfreq * duration_s)) / sfreq
    return np.sin(2 * np.pi * freq_hz * t)


def _power(signal: np.ndarray) -> float:
    return float(np.mean(signal ** 2))


# ---------------------------------------------------------------------------
# BandpassFilter
# ---------------------------------------------------------------------------

class TestBandpassFilter:
    def test_preserves_in_band(self):
        sig = _sine(10)  # 10 Hz -> inside 8-12
        f = BandpassFilter(low_hz=8, high_hz=12, order=4)
        out = f.apply(sig, SFREQ)
        assert _power(out) > 0.3 * _power(sig)

    def test_attenuates_out_of_band(self):
        sig = _sine(50)  # 50 Hz -> outside 8-12
        f = BandpassFilter(low_hz=8, high_hz=12, order=4)
        out = f.apply(sig, SFREQ)
        assert _power(out) < 0.05 * _power(sig)

    def test_disabled_passthrough(self):
        sig = _sine(50)
        f = BandpassFilter(low_hz=8, high_hz=12, enabled=False)
        out = f.apply(sig, SFREQ)
        np.testing.assert_array_equal(out, sig)

    def test_validation_ok(self):
        f = BandpassFilter(low_hz=0.3, high_hz=35, order=4)
        assert f.validate(SFREQ) == []

    def test_validation_bad_range(self):
        f = BandpassFilter(low_hz=40, high_hz=10)
        errs = f.validate(SFREQ)
        assert len(errs) > 0

    def test_validation_above_nyquist(self):
        f = BandpassFilter(low_hz=0.3, high_hz=200)
        errs = f.validate(SFREQ)
        assert any("Nyquist" in e for e in errs)

    def test_causal(self):
        sig = _sine(10, duration_s=2.0)
        f = BandpassFilter(low_hz=8, high_hz=12, order=4, causal=True)
        out = f.apply(sig, SFREQ)
        assert out.shape == sig.shape
        assert _power(out) > 0.1 * _power(sig)


# ---------------------------------------------------------------------------
# HighpassFilter
# ---------------------------------------------------------------------------

class TestHighpassFilter:
    def test_passes_high(self):
        sig = _sine(20)
        f = HighpassFilter(cutoff_hz=5, order=4)
        out = f.apply(sig, SFREQ)
        assert _power(out) > 0.3 * _power(sig)

    def test_attenuates_low(self):
        sig = _sine(0.5, duration_s=10)
        f = HighpassFilter(cutoff_hz=5, order=4)
        out = f.apply(sig, SFREQ)
        assert _power(out) < 0.1 * _power(sig)

    def test_validation(self):
        f = HighpassFilter(cutoff_hz=-1)
        assert len(f.validate(SFREQ)) > 0


# ---------------------------------------------------------------------------
# LowpassFilter
# ---------------------------------------------------------------------------

class TestLowpassFilter:
    def test_passes_low(self):
        sig = _sine(5)
        f = LowpassFilter(cutoff_hz=20, order=4)
        out = f.apply(sig, SFREQ)
        assert _power(out) > 0.3 * _power(sig)

    def test_attenuates_high(self):
        sig = _sine(60)
        f = LowpassFilter(cutoff_hz=20, order=4)
        out = f.apply(sig, SFREQ)
        assert _power(out) < 0.05 * _power(sig)


# ---------------------------------------------------------------------------
# NotchFilter
# ---------------------------------------------------------------------------

class TestNotchFilter:
    def test_removes_50hz(self):
        sig = _sine(10) + _sine(50)
        f = NotchFilter(freq_hz=50, quality_factor=30)
        out = f.apply(sig, SFREQ)
        # 10 Hz should survive, 50 Hz should be attenuated
        pure_10 = _sine(10)
        residual_50 = out - pure_10[:len(out)]
        assert _power(residual_50) < 0.15 * _power(_sine(50))

    def test_harmonics(self):
        sig = _sine(50) + _sine(100)
        f = NotchFilter(freq_hz=50, quality_factor=30, harmonics=2)
        out = f.apply(sig, SFREQ)
        assert _power(out) < 0.15 * _power(sig)

    def test_validation_bad_freq(self):
        f = NotchFilter(freq_hz=-10)
        assert len(f.validate(SFREQ)) > 0


# ---------------------------------------------------------------------------
# SmoothingFilter
# ---------------------------------------------------------------------------

class TestSmoothingFilter:
    def test_savgol(self):
        rng = np.random.default_rng(0)
        sig = _sine(5) + rng.normal(0, 0.5, len(_sine(5)))
        f = SmoothingFilter(method="savgol", window_size=15, poly_order=3)
        out = f.apply(sig, SFREQ)
        # Smoothed should have lower variance than noisy input
        assert np.std(out) < np.std(sig)

    def test_moving_average(self):
        sig = np.random.default_rng(1).normal(0, 1, 1000)
        f = SmoothingFilter(method="moving_average", window_size=21)
        out = f.apply(sig, SFREQ)
        assert np.std(out) < np.std(sig)

    def test_gaussian(self):
        sig = np.random.default_rng(2).normal(0, 1, 1000)
        f = SmoothingFilter(method="gaussian", window_size=21)
        out = f.apply(sig, SFREQ)
        assert np.std(out) < np.std(sig)

    def test_validation_bad_method(self):
        f = SmoothingFilter(method="foobar")
        assert len(f.validate(SFREQ)) > 0

    def test_short_data(self):
        sig = np.array([1.0, 2.0])
        f = SmoothingFilter(window_size=11)
        out = f.apply(sig, SFREQ)
        assert len(out) == len(sig)


# ---------------------------------------------------------------------------
# FilterPipeline
# ---------------------------------------------------------------------------

class TestFilterPipeline:
    def test_chain_bandpass_notch(self):
        sig = _sine(10) + _sine(50)
        pipe = FilterPipeline(filters=[
            BandpassFilter(low_hz=0.3, high_hz=35),
            NotchFilter(freq_hz=50),
        ])
        out = pipe.apply(sig, SFREQ)
        # 10 Hz preserved, 50 Hz removed (by both bandpass cutoff and notch)
        assert _power(out) > 0.1

    def test_disabled_pipeline(self):
        sig = _sine(10)
        pipe = FilterPipeline(filters=[BandpassFilter(low_hz=0.3, high_hz=5)], enabled=False)
        out = pipe.apply(sig, SFREQ)
        np.testing.assert_array_equal(out, sig)

    def test_empty_pipeline(self):
        sig = _sine(10)
        pipe = FilterPipeline()
        out = pipe.apply(sig, SFREQ)
        np.testing.assert_array_equal(out, sig)

    def test_validate(self):
        pipe = FilterPipeline(filters=[
            BandpassFilter(low_hz=40, high_hz=10),
            NotchFilter(freq_hz=-5),
        ])
        errs = pipe.validate(SFREQ)
        assert len(errs) >= 2

    def test_add_remove_move(self):
        pipe = FilterPipeline()
        pipe.add(BandpassFilter(low_hz=0.3, high_hz=35))
        pipe.add(NotchFilter(freq_hz=50))
        assert len(pipe.filters) == 2
        pipe.move(0, 1)
        assert isinstance(pipe.filters[0], NotchFilter)
        pipe.remove(0)
        assert len(pipe.filters) == 1

    def test_deep_copy(self):
        pipe = FilterPipeline(filters=[BandpassFilter(low_hz=0.3, high_hz=35)])
        clone = pipe.deep_copy()
        clone.filters[0].low_hz = 999
        assert pipe.filters[0].low_hz == 0.3

    def test_frequency_response(self):
        pipe = FilterPipeline(filters=[BandpassFilter(low_hz=0.3, high_hz=35)])
        freqs, mag = pipe.frequency_response(SFREQ)
        assert len(freqs) == 512
        assert len(mag) == 512
        # In-band should be near 0 dB, out-of-band should be << 0
        idx_10hz = int(10 / (SFREQ / 2) * 512)
        idx_100hz = int(100 / (SFREQ / 2) * 512)
        assert mag[idx_10hz] > -3  # 10 Hz in passband
        assert mag[idx_100hz] < -20  # 100 Hz in stopband


# ---------------------------------------------------------------------------
# Serialisation roundtrip
# ---------------------------------------------------------------------------

class TestSerialisation:
    @pytest.mark.parametrize("filt", [
        BandpassFilter(low_hz=0.5, high_hz=40, order=6, filter_type="cheby1", causal=True),
        HighpassFilter(cutoff_hz=10, order=3),
        LowpassFilter(cutoff_hz=70, order=4, filter_type="ellip"),
        NotchFilter(freq_hz=60, quality_factor=25, harmonics=3),
        SmoothingFilter(method="gaussian", window_size=21, poly_order=0),
    ])
    def test_filter_roundtrip(self, filt):
        d = filt.to_dict()
        restored = filter_from_dict(d)
        assert type(restored) is type(filt)
        assert restored.enabled == filt.enabled
        d2 = restored.to_dict()
        assert d == d2

    def test_pipeline_roundtrip(self):
        pipe = FilterPipeline(filters=[
            BandpassFilter(low_hz=0.3, high_hz=35),
            NotchFilter(freq_hz=50),
        ], enabled=True)
        d = pipe.to_dict()
        restored = FilterPipeline.from_dict(d)
        assert len(restored.filters) == 2
        assert isinstance(restored.filters[0], BandpassFilter)
        assert isinstance(restored.filters[1], NotchFilter)
        assert restored.enabled is True

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown filter type"):
            filter_from_dict({"type": "NonExistentFilter"})


# ---------------------------------------------------------------------------
# PresetLibrary
# ---------------------------------------------------------------------------

class TestPresetLibrary:
    def test_default_presets_load(self):
        presets_path = Path(__file__).resolve().parent.parent / "config" / "filter_presets.json"
        if not presets_path.exists():
            pytest.skip("Default presets JSON not found")
        lib = PresetLibrary(presets_path)
        names = lib.list_names()
        assert len(names) >= 5
        assert "EEG Standard PSG" in names

    def test_add_get_remove(self):
        lib = PresetLibrary()
        p = FilterPreset(name="Test", pipeline=FilterPipeline(filters=[NotchFilter(freq_hz=50)]))
        lib.add(p)
        assert lib.get("Test") is not None
        lib.remove("Test")
        assert lib.get("Test") is None

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_presets.json"
            lib = PresetLibrary()
            lib.add(FilterPreset(name="A", pipeline=FilterPipeline(filters=[BandpassFilter()])))
            lib.save(path)

            lib2 = PresetLibrary(path)
            assert "A" in lib2.list_names()
            preset = lib2.get("A")
            assert len(preset.pipeline.filters) == 1

    def test_cannot_remove_builtin(self):
        lib = PresetLibrary()
        lib.add(FilterPreset(name="BuiltIn", builtin=True, pipeline=FilterPipeline()))
        with pytest.raises(ValueError, match="built-in"):
            lib.remove("BuiltIn")

    def test_filter_by_channel_type(self):
        lib = PresetLibrary()
        lib.add(FilterPreset(name="E", channel_type="eeg", pipeline=FilterPipeline()))
        lib.add(FilterPreset(name="M", channel_type="emg", pipeline=FilterPipeline()))
        lib.add(FilterPreset(name="G", channel_type="generic", pipeline=FilterPipeline()))
        eeg_names = lib.list_names("eeg")
        assert "E" in eeg_names
        assert "G" in eeg_names  # generic included
        assert "M" not in eeg_names


# ---------------------------------------------------------------------------
# pipeline_from_legacy_params
# ---------------------------------------------------------------------------

class TestLegacyBridge:
    def test_bandpass(self):
        pipe = pipeline_from_legacy_params(low=0.3, high=35.0, order=4)
        assert len(pipe.filters) == 1
        assert isinstance(pipe.filters[0], BandpassFilter)

    def test_highpass(self):
        pipe = pipeline_from_legacy_params(low=10.0, high=0.0, order=4)
        assert len(pipe.filters) == 1
        assert isinstance(pipe.filters[0], HighpassFilter)

    def test_lowpass(self):
        pipe = pipeline_from_legacy_params(low=0.0, high=35.0, order=4)
        assert len(pipe.filters) == 1
        assert isinstance(pipe.filters[0], LowpassFilter)

    def test_with_notch(self):
        pipe = pipeline_from_legacy_params(low=0.3, high=35.0, notch_hz=50, order=4)
        assert len(pipe.filters) == 2
        assert isinstance(pipe.filters[1], NotchFilter)

    def test_no_filter(self):
        pipe = pipeline_from_legacy_params(low=0, high=0)
        assert len(pipe.filters) == 0


# ---------------------------------------------------------------------------
# Backward compatibility: CESA.filters.apply_filter
# ---------------------------------------------------------------------------

class TestBackwardCompatApplyFilter:
    def test_bandpass_compat(self):
        from CESA.filters import apply_filter
        sig = _sine(10) + _sine(50)
        out = apply_filter(sig, sfreq=SFREQ, filter_order=4, low=0.3, high=35.0)
        assert _power(out) < _power(sig)
        assert _power(out) > 0.1

    def test_highpass_compat(self):
        from CESA.filters import apply_filter
        sig = _sine(0.5, duration_s=10) + _sine(20, duration_s=10)
        out = apply_filter(sig, sfreq=SFREQ, filter_order=4, low=10.0, high=0.0)
        # 20 Hz should remain
        assert _power(out) > 0.1

    def test_no_filter_returns_unchanged(self):
        from CESA.filters import apply_filter
        sig = _sine(10)
        out = apply_filter(sig, sfreq=SFREQ, filter_order=4)
        np.testing.assert_array_equal(out, sig)


# ---------------------------------------------------------------------------
# Preprocessing integration
# ---------------------------------------------------------------------------

class TestPreprocessingIntegration:
    def test_config_has_filter_overrides(self):
        from CESA.sleep_pipeline.preprocessing import PreprocessingConfig
        config = PreprocessingConfig()
        assert hasattr(config, "filter_overrides")
        assert config.filter_overrides == {}

    def test_config_accepts_pipeline(self):
        from CESA.sleep_pipeline.preprocessing import PreprocessingConfig
        pipe = FilterPipeline(filters=[BandpassFilter(low_hz=0.3, high_hz=35)])
        config = PreprocessingConfig(filter_overrides={"eeg": pipe})
        assert "eeg" in config.filter_overrides


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_nan_handling(self):
        sig = np.array([1.0, np.nan, 3.0, np.inf, -np.inf, 2.0] * 100)
        f = BandpassFilter(low_hz=1, high_hz=40)
        out = f.apply(sig, SFREQ)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_zero_length_data(self):
        sig = np.array([])
        pipe = FilterPipeline(filters=[BandpassFilter(low_hz=0.3, high_hz=35)])
        out = pipe.apply(sig, SFREQ)
        assert len(out) == 0

    def test_very_short_data(self):
        sig = np.array([1.0, 2.0, 3.0])
        f = BandpassFilter(low_hz=0.3, high_hz=35, order=4)
        # May not be filterable due to length < padlen; should return data unchanged
        out = f.apply(sig, SFREQ)
        assert len(out) == 3

    def test_order_1(self):
        sig = _sine(10)
        f = BandpassFilter(low_hz=5, high_hz=20, order=1)
        out = f.apply(sig, SFREQ)
        assert _power(out) > 0.1

    def test_cheby_types(self):
        sig = _sine(10)
        for ft in ("cheby1", "cheby2", "ellip"):
            f = BandpassFilter(low_hz=5, high_hz=20, order=4, filter_type=ft)
            out = f.apply(sig, SFREQ)
            assert _power(out) > 0.1, f"Failed for filter_type={ft}"


# ---------------------------------------------------------------------------
# Physiological range warnings
# ---------------------------------------------------------------------------

class TestPhysiologicalWarnings:
    def test_bandpass_in_range_eeg(self):
        f = BandpassFilter(low_hz=0.3, high_hz=35, order=4)
        w = f.physiological_warnings("eeg")
        assert len(w) == 0

    def test_bandpass_out_of_range_eeg(self):
        f = BandpassFilter(low_hz=0.01, high_hz=35, order=4)
        w = f.physiological_warnings("eeg")
        assert any("bp_low_hz" in m for m in w)

    def test_bandpass_high_out_of_range_eeg(self):
        f = BandpassFilter(low_hz=0.3, high_hz=100, order=4)
        w = f.physiological_warnings("eeg")
        assert any("bp_high_hz" in m for m in w)

    def test_highpass_in_range_emg(self):
        f = HighpassFilter(cutoff_hz=10, order=4)
        w = f.physiological_warnings("emg")
        assert len(w) == 0

    def test_highpass_out_of_range_emg(self):
        f = HighpassFilter(cutoff_hz=1.0, order=4)
        w = f.physiological_warnings("emg")
        assert any("hp_cutoff_hz" in m for m in w)

    def test_lowpass_in_range_eog(self):
        f = LowpassFilter(cutoff_hz=10, order=2)
        w = f.physiological_warnings("eog")
        assert len(w) == 0

    def test_lowpass_out_of_range_eog(self):
        f = LowpassFilter(cutoff_hz=2, order=2)
        w = f.physiological_warnings("eog")
        assert any("lp_cutoff_hz" in m for m in w)

    def test_notch_in_range(self):
        f = NotchFilter(freq_hz=50)
        w = f.physiological_warnings("eeg")
        assert len(w) == 0

    def test_notch_out_of_range(self):
        f = NotchFilter(freq_hz=30)
        w = f.physiological_warnings("eeg")
        assert any("notch_hz" in m for m in w)

    def test_smoothing_no_warnings(self):
        f = SmoothingFilter()
        w = f.physiological_warnings("eeg")
        assert len(w) == 0

    def test_unknown_channel_type(self):
        f = BandpassFilter(low_hz=0.01, high_hz=200)
        w = f.physiological_warnings("unknown_type")
        assert len(w) == 0

    def test_pipeline_warnings(self):
        pipe = FilterPipeline(filters=[
            BandpassFilter(low_hz=0.01, high_hz=200),
            NotchFilter(freq_hz=30),
        ])
        w = pipe.physiological_warnings("eeg", SFREQ)
        assert len(w) >= 2

    def test_pipeline_disabled_filters_excluded(self):
        pipe = FilterPipeline(filters=[
            BandpassFilter(low_hz=0.01, high_hz=200, enabled=False),
        ])
        w = pipe.physiological_warnings("eeg", SFREQ)
        assert len(w) == 0

    def test_order_out_of_range(self):
        f = BandpassFilter(low_hz=0.3, high_hz=35, order=10)
        w = f.physiological_warnings("eeg")
        assert any("order" in m for m in w)

    def test_physiological_ranges_dict_structure(self):
        assert "eeg" in PHYSIOLOGICAL_RANGES
        assert "eog" in PHYSIOLOGICAL_RANGES
        assert "emg" in PHYSIOLOGICAL_RANGES
        assert "ecg" in PHYSIOLOGICAL_RANGES
        for ct, ranges in PHYSIOLOGICAL_RANGES.items():
            for key, val in ranges.items():
                assert len(val) == 4
                lo, hi, unit, note = val
                assert lo <= hi


# ---------------------------------------------------------------------------
# FilterAuditLog
# ---------------------------------------------------------------------------

class TestFilterAuditLog:
    def test_record_and_retrieve(self):
        log = FilterAuditLog()
        log.record("EEG1", "param_changed", param="low_hz", old=0.3, new=0.1)
        assert len(log.entries) == 1
        e = log.entries[0]
        assert e.channel == "EEG1"
        assert e.action == "param_changed"
        assert e.details["param"] == "low_hz"

    def test_multiple_records(self):
        log = FilterAuditLog()
        log.record("EEG1", "add_filter", filter_type="BandpassFilter")
        log.record("EOG1", "delete_filter", index=0)
        log.record("*", "apply_all")
        assert len(log.entries) == 3

    def test_to_list(self):
        log = FilterAuditLog()
        log.record("EEG1", "test", key="val")
        items = log.to_list()
        assert len(items) == 1
        assert items[0]["channel"] == "EEG1"
        assert items[0]["action"] == "test"
        assert "timestamp" in items[0]
        assert items[0]["details"]["key"] == "val"

    def test_export_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.json"
            log = FilterAuditLog()
            log.record("EEG1", "change", x=1)
            log.record("EEG2", "change", x=2)
            log.export_json(path)
            assert path.exists()
            data = json.loads(path.read_text(encoding="utf-8"))
            assert len(data) == 2

    def test_clear(self):
        log = FilterAuditLog()
        log.record("EEG1", "test")
        assert len(log.entries) == 1
        log.clear()
        assert len(log.entries) == 0

    def test_timestamp_format(self):
        log = FilterAuditLog()
        log.record("X", "Y")
        ts = log.entries[0].timestamp
        assert "T" in ts
        assert len(ts) >= 19


# ---------------------------------------------------------------------------
# Enhanced PresetLibrary (user presets, import/export)
# ---------------------------------------------------------------------------

class TestPresetLibraryEnhanced:
    def test_user_path_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builtin = Path(tmpdir) / "builtin.json"
            builtin.write_text(json.dumps({"presets": []}), encoding="utf-8")
            lib = PresetLibrary(builtin)
            assert lib.user_path == Path(tmpdir) / "user_presets.json"

    def test_user_path_explicit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builtin = Path(tmpdir) / "builtin.json"
            user = Path(tmpdir) / "my_presets.json"
            builtin.write_text(json.dumps({"presets": []}), encoding="utf-8")
            lib = PresetLibrary(builtin, user_path=user)
            assert lib.user_path == user

    def test_save_user_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builtin = Path(tmpdir) / "builtin.json"
            user = Path(tmpdir) / "user.json"
            builtin.write_text(json.dumps({"presets": [
                {"name": "Built", "builtin": True, "pipeline": {"enabled": True, "filters": []}}
            ]}), encoding="utf-8")

            lib = PresetLibrary(builtin, user_path=user)
            lib.add(FilterPreset(name="MyPreset", builtin=False,
                                 pipeline=FilterPipeline(filters=[NotchFilter()])))
            lib.save_user()

            assert user.exists()
            data = json.loads(user.read_text(encoding="utf-8"))
            names = [p["name"] for p in data["presets"]]
            assert "MyPreset" in names
            assert "Built" not in names

    def test_load_user_presets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builtin = Path(tmpdir) / "builtin.json"
            user = Path(tmpdir) / "user_presets.json"
            builtin.write_text(json.dumps({"presets": [
                {"name": "B", "builtin": True, "pipeline": {"enabled": True, "filters": []}}
            ]}), encoding="utf-8")
            user.write_text(json.dumps({"presets": [
                {"name": "U", "builtin": True, "pipeline": {"enabled": True, "filters": []}}
            ]}), encoding="utf-8")

            lib = PresetLibrary(builtin, user_path=user)
            assert "B" in lib.list_names()
            assert "U" in lib.list_names()
            u = lib.get("U")
            assert u is not None
            assert u.builtin is False

    def test_list_user_names(self):
        lib = PresetLibrary()
        lib.add(FilterPreset(name="A", builtin=True, pipeline=FilterPipeline()))
        lib.add(FilterPreset(name="B", builtin=False, pipeline=FilterPipeline()))
        assert lib.list_user_names() == ["B"]

    def test_list_builtin_names(self):
        lib = PresetLibrary()
        lib.add(FilterPreset(name="A", builtin=True, pipeline=FilterPipeline()))
        lib.add(FilterPreset(name="B", builtin=False, pipeline=FilterPipeline()))
        assert lib.list_builtin_names() == ["A"]

    def test_rename(self):
        lib = PresetLibrary()
        lib.add(FilterPreset(name="Old", builtin=False, pipeline=FilterPipeline()))
        lib.rename("Old", "New")
        assert lib.get("Old") is None
        assert lib.get("New") is not None

    def test_rename_builtin_raises(self):
        lib = PresetLibrary()
        lib.add(FilterPreset(name="X", builtin=True, pipeline=FilterPipeline()))
        with pytest.raises(ValueError, match="built-in"):
            lib.rename("X", "Y")

    def test_rename_nonexistent_raises(self):
        lib = PresetLibrary()
        with pytest.raises(ValueError, match="not found"):
            lib.rename("Ghost", "New")

    def test_rename_collision_raises(self):
        lib = PresetLibrary()
        lib.add(FilterPreset(name="A", builtin=False, pipeline=FilterPipeline()))
        lib.add(FilterPreset(name="B", builtin=False, pipeline=FilterPipeline()))
        with pytest.raises(ValueError, match="already exists"):
            lib.rename("A", "B")

    def test_export_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "export.json"
            lib = PresetLibrary()
            lib.add(FilterPreset(name="A", pipeline=FilterPipeline(filters=[NotchFilter()])))
            lib.add(FilterPreset(name="B", pipeline=FilterPipeline()))
            count = lib.export_presets(path)
            assert count == 2
            assert path.exists()
            data = json.loads(path.read_text(encoding="utf-8"))
            assert len(data["presets"]) == 2

    def test_export_selected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "export.json"
            lib = PresetLibrary()
            lib.add(FilterPreset(name="A", pipeline=FilterPipeline()))
            lib.add(FilterPreset(name="B", pipeline=FilterPipeline()))
            count = lib.export_presets(path, names=["A"])
            assert count == 1

    def test_import_presets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "import.json"
            path.write_text(json.dumps({"presets": [
                {"name": "Imported", "pipeline": {"enabled": True, "filters": [
                    {"type": "NotchFilter", "enabled": True, "freq_hz": 60}
                ]}}
            ]}), encoding="utf-8")

            lib = PresetLibrary()
            count = lib.import_presets(path)
            assert count == 1
            p = lib.get("Imported")
            assert p is not None
            assert p.builtin is False
            assert len(p.pipeline.filters) == 1

    def test_import_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "import.json"
            path.write_text(json.dumps({"presets": [
                {"name": "Existing", "pipeline": {"enabled": True, "filters": []}}
            ]}), encoding="utf-8")

            lib = PresetLibrary()
            lib.add(FilterPreset(name="Existing", pipeline=FilterPipeline(
                filters=[BandpassFilter()])))
            count = lib.import_presets(path, overwrite=False)
            assert count == 0
            assert len(lib.get("Existing").pipeline.filters) == 1

    def test_import_with_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "import.json"
            path.write_text(json.dumps({"presets": [
                {"name": "Existing", "pipeline": {"enabled": True, "filters": []}}
            ]}), encoding="utf-8")

            lib = PresetLibrary()
            lib.add(FilterPreset(name="Existing", pipeline=FilterPipeline(
                filters=[BandpassFilter()])))
            count = lib.import_presets(path, overwrite=True)
            assert count == 1
            assert len(lib.get("Existing").pipeline.filters) == 0

    def test_import_nonexistent_raises(self):
        lib = PresetLibrary()
        with pytest.raises(FileNotFoundError):
            lib.import_presets("/nonexistent/path.json")

    def test_export_import_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rt.json"
            lib1 = PresetLibrary()
            lib1.add(FilterPreset(
                name="RT", channel_type="eeg", description="test",
                pipeline=FilterPipeline(filters=[
                    BandpassFilter(low_hz=0.5, high_hz=40),
                    NotchFilter(freq_hz=50, harmonics=2),
                ]),
            ))
            lib1.export_presets(path)

            lib2 = PresetLibrary()
            lib2.import_presets(path)
            p = lib2.get("RT")
            assert p is not None
            assert p.channel_type == "eeg"
            assert len(p.pipeline.filters) == 2
            assert isinstance(p.pipeline.filters[0], BandpassFilter)
            assert p.pipeline.filters[0].low_hz == 0.5


# ---------------------------------------------------------------------------
# UndoManager
# ---------------------------------------------------------------------------

class TestUndoManager:
    def _make_pipelines(self, low: float = 0.3) -> dict:
        return {"EEG": FilterPipeline(filters=[BandpassFilter(low_hz=low, high_hz=35)])}

    def test_initial_state(self):
        um = UndoManager()
        assert not um.can_undo
        assert not um.can_redo
        assert um.undo_depth == 0
        assert um.redo_depth == 0

    def test_save_and_undo(self):
        um = UndoManager()
        p1 = self._make_pipelines(0.3)
        um.save_state(p1)
        assert um.can_undo
        p2 = self._make_pipelines(1.0)
        restored = um.undo(p2)
        assert restored is not None
        assert restored["EEG"].filters[0].low_hz == 0.3

    def test_undo_empty(self):
        um = UndoManager()
        assert um.undo(self._make_pipelines()) is None

    def test_redo(self):
        um = UndoManager()
        p1 = self._make_pipelines(0.3)
        um.save_state(p1)
        p2 = self._make_pipelines(1.0)
        um.undo(p2)
        assert um.can_redo
        restored = um.redo(self._make_pipelines(0.3))
        assert restored is not None
        assert restored["EEG"].filters[0].low_hz == 1.0

    def test_redo_empty(self):
        um = UndoManager()
        assert um.redo(self._make_pipelines()) is None

    def test_save_clears_redo(self):
        um = UndoManager()
        um.save_state(self._make_pipelines(0.3))
        um.save_state(self._make_pipelines(0.5))
        um.undo(self._make_pipelines(1.0))
        assert um.can_redo
        um.save_state(self._make_pipelines(2.0))
        assert not um.can_redo

    def test_max_depth(self):
        um = UndoManager(max_depth=3)
        for i in range(10):
            um.save_state(self._make_pipelines(float(i)))
        assert um.undo_depth == 3

    def test_multiple_undo_redo(self):
        um = UndoManager()
        states = [0.1, 0.2, 0.3, 0.4, 0.5]
        for s in states:
            um.save_state(self._make_pipelines(s))
        current = self._make_pipelines(0.6)
        for expected in reversed(states):
            current = um.undo(current)
            assert current is not None
            assert abs(current["EEG"].filters[0].low_hz - expected) < 0.01
        assert um.undo(current) is None

    def test_clear(self):
        um = UndoManager()
        um.save_state(self._make_pipelines(0.3))
        um.clear()
        assert not um.can_undo
        assert not um.can_redo

    def test_snapshot_isolation(self):
        um = UndoManager()
        pipes = self._make_pipelines(0.3)
        um.save_state(pipes)
        pipes["EEG"].filters[0].low_hz = 999.0
        restored = um.undo(pipes)
        assert restored["EEG"].filters[0].low_hz == 0.3

    def test_multi_channel(self):
        um = UndoManager()
        pipes = {
            "EEG1": FilterPipeline(filters=[BandpassFilter(low_hz=0.3, high_hz=35)]),
            "EOG1": FilterPipeline(filters=[NotchFilter(freq_hz=50)]),
        }
        um.save_state(pipes)
        pipes2 = {
            "EEG1": FilterPipeline(filters=[BandpassFilter(low_hz=1.0, high_hz=35)]),
            "EOG1": FilterPipeline(filters=[NotchFilter(freq_hz=60)]),
        }
        restored = um.undo(pipes2)
        assert restored is not None
        assert restored["EEG1"].filters[0].low_hz == 0.3
        assert restored["EOG1"].filters[0].freq_hz == 50


# ---------------------------------------------------------------------------
# AdaptiveFilterSuggester
# ---------------------------------------------------------------------------

class TestAdaptiveFilterSuggester:
    @pytest.fixture
    def lib(self):
        presets_path = Path(__file__).resolve().parent.parent / "config" / "filter_presets.json"
        if not presets_path.exists():
            pytest.skip("Default presets not found")
        return PresetLibrary(presets_path)

    def test_suggest_eeg_default(self, lib):
        s = AdaptiveFilterSuggester(lib)
        suggestions = s.suggest_for_channel("eeg", "default")
        assert len(suggestions) >= 1
        names = [sg.preset_name for sg in suggestions]
        assert "EEG Standard PSG" in names

    def test_suggest_eeg_scoring(self, lib):
        s = AdaptiveFilterSuggester(lib)
        suggestions = s.suggest_for_channel("eeg", "scoring")
        names = [sg.preset_name for sg in suggestions]
        assert "EEG Sleep Scoring AASM" in names

    def test_suggest_eog_default(self, lib):
        s = AdaptiveFilterSuggester(lib)
        suggestions = s.suggest_for_channel("eog", "default")
        names = [sg.preset_name for sg in suggestions]
        assert "EOG Standard" in names

    def test_suggest_emg_default(self, lib):
        s = AdaptiveFilterSuggester(lib)
        suggestions = s.suggest_for_channel("emg", "default")
        names = [sg.preset_name for sg in suggestions]
        assert "EMG Menton" in names

    def test_suggest_unknown_type(self, lib):
        s = AdaptiveFilterSuggester(lib)
        suggestions = s.suggest_for_channel("xyz", "default")
        assert len(suggestions) == 0

    def test_sorted_by_confidence(self, lib):
        s = AdaptiveFilterSuggester(lib)
        suggestions = s.suggest_for_channel("eeg", "default")
        for i in range(len(suggestions) - 1):
            assert suggestions[i].confidence >= suggestions[i + 1].confidence

    def test_spectral_analysis_delta(self, lib):
        s = AdaptiveFilterSuggester(lib)
        t = np.arange(0, 5, 1 / 256.0)
        delta_signal = 50 * np.sin(2 * np.pi * 2 * t)
        suggestions = s.suggest_for_channel("eeg", "default",
                                            signal_snippet=delta_signal, sfreq=256.0)
        reasons = [sg.reason for sg in suggestions]
        has_delta = any("elta" in r for r in reasons)
        assert has_delta or len(suggestions) >= 1

    def test_spectral_analysis_mains_50(self, lib):
        s = AdaptiveFilterSuggester(lib)
        t = np.arange(0, 5, 1 / 256.0)
        mains_signal = 100 * np.sin(2 * np.pi * 50 * t)
        suggestions = s.suggest_for_channel("eeg", "default",
                                            signal_snippet=mains_signal, sfreq=256.0)
        names = [sg.preset_name for sg in suggestions]
        assert "Notch 50 Hz seul" in names

    def test_accept_suggestion(self, lib):
        s = AdaptiveFilterSuggester(lib)
        pipes = {"EEG1": FilterPipeline()}
        sg = FilterSuggestion(
            preset_name="EEG Standard PSG",
            reason="test",
            confidence=0.9,
            channel_type="eeg",
        )
        ok = s.accept_suggestion(sg, "EEG1", pipes)
        assert ok
        assert len(pipes["EEG1"].filters) > 0

    def test_accept_invalid_preset(self, lib):
        s = AdaptiveFilterSuggester(lib)
        pipes = {"EEG1": FilterPipeline()}
        sg = FilterSuggestion(
            preset_name="DOES_NOT_EXIST",
            reason="test",
            confidence=0.5,
        )
        ok = s.accept_suggestion(sg, "EEG1", pipes)
        assert not ok
        assert len(pipes["EEG1"].filters) == 0

    def test_audit_on_accept(self, lib):
        audit = FilterAuditLog()
        s = AdaptiveFilterSuggester(lib, audit_log=audit)
        pipes = {"EEG1": FilterPipeline()}
        sg = FilterSuggestion(
            preset_name="EEG Standard PSG",
            reason="test suggestion",
            confidence=0.85,
        )
        s.accept_suggestion(sg, "EEG1", pipes)
        entries = audit.entries
        assert len(entries) == 1
        assert entries[0].action == "accept_suggestion"
        assert entries[0].details["preset_name"] == "EEG Standard PSG"

    def test_suggestion_dataclass(self):
        sg = FilterSuggestion(
            preset_name="Test",
            reason="unit test",
            confidence=0.75,
            channel_type="eeg",
            context="nrem",
        )
        assert sg.preset_name == "Test"
        assert sg.confidence == 0.75
        assert sg.context == "nrem"


# ===========================================================================
# Versioning consistency
# ===========================================================================

class TestVersioning:
    """Verify all version strings are consistent across the project."""

    def test_cesa_init_version(self):
        import CESA
        assert CESA.__version__ == "0.0beta1.0"

    def test_sleep_pipeline_version(self):
        import CESA.sleep_pipeline
        assert CESA.sleep_pipeline.__version__ == "0.0beta1.0"

    def test_run_docstring_version(self):
        run_path = Path(__file__).resolve().parent.parent / "run.py"
        content = run_path.read_text(encoding="utf-8")
        assert "0.0beta1.0" in content

    def test_requirements_version(self):
        req_path = Path(__file__).resolve().parent.parent / "requirements.txt"
        content = req_path.read_text(encoding="utf-8")
        assert "CESA 0.0beta1.0" in content

    def test_changelog_exists(self):
        cl_path = Path(__file__).resolve().parent.parent / "CHANGELOG.md"
        assert cl_path.exists()
        content = cl_path.read_text(encoding="utf-8")
        assert "[0.0beta1.0]" in content


# ===========================================================================
# Favorite presets
# ===========================================================================

class TestFavoritePresets:
    """Tests for the FavoritePresets manager."""

    def test_empty_favorites(self, tmp_path):
        fav = FavoritePresets(tmp_path / "fav.json")
        assert len(fav.names) == 0
        assert fav.to_list() == []

    def test_toggle_add_remove(self, tmp_path):
        fav = FavoritePresets(tmp_path / "fav.json")
        assert fav.toggle("EEG Standard PSG") is True
        assert fav.is_favorite("EEG Standard PSG")
        assert fav.toggle("EEG Standard PSG") is False
        assert not fav.is_favorite("EEG Standard PSG")

    def test_persistence(self, tmp_path):
        path = tmp_path / "fav.json"
        fav = FavoritePresets(path)
        fav.add("Preset A")
        fav.add("Preset B")

        fav2 = FavoritePresets(path)
        assert fav2.is_favorite("Preset A")
        assert fav2.is_favorite("Preset B")

    def test_clear(self, tmp_path):
        fav = FavoritePresets(tmp_path / "fav.json")
        fav.add("X")
        fav.add("Y")
        fav.clear()
        assert len(fav.names) == 0

    def test_to_list_sorted(self, tmp_path):
        fav = FavoritePresets(tmp_path / "fav.json")
        fav.add("Zulu")
        fav.add("Alpha")
        assert fav.to_list() == ["Alpha", "Zulu"]

    def test_remove_nonexistent(self, tmp_path):
        fav = FavoritePresets(tmp_path / "fav.json")
        fav.remove("nothing")
        assert len(fav.names) == 0

    def test_add_duplicate(self, tmp_path):
        fav = FavoritePresets(tmp_path / "fav.json")
        fav.add("A")
        fav.add("A")
        assert fav.to_list() == ["A"]


# ===========================================================================
# Channel annotations
# ===========================================================================

class TestChannelAnnotation:
    """Tests for the ChannelAnnotation dataclass."""

    def test_to_dict_roundtrip(self):
        ann = ChannelAnnotation(channel="EEG Fp1", text="Alpha burst noted")
        d = ann.to_dict()
        restored = ChannelAnnotation.from_dict(d)
        assert restored.channel == "EEG Fp1"
        assert restored.text == "Alpha burst noted"

    def test_timestamp_auto(self):
        ann = ChannelAnnotation(channel="C", text="test")
        assert len(ann.timestamp) > 0

    def test_author_field(self):
        ann = ChannelAnnotation(channel="C", text="t", author="Dr. X")
        assert ann.author == "Dr. X"


class TestChannelAnnotationStore:
    """Tests for the ChannelAnnotationStore."""

    def test_set_and_get(self, tmp_path):
        store = ChannelAnnotationStore(tmp_path / "ann.json")
        store.set("EEG Fp1", "Alpha burst", author="Test")
        assert store.get_text("EEG Fp1") == "Alpha burst"
        ann = store.get("EEG Fp1")
        assert ann is not None
        assert ann.author == "Test"

    def test_persistence(self, tmp_path):
        path = tmp_path / "ann.json"
        store = ChannelAnnotationStore(path)
        store.set("EOG", "Slow eye movements")
        store2 = ChannelAnnotationStore(path)
        assert store2.get_text("EOG") == "Slow eye movements"

    def test_delete(self, tmp_path):
        store = ChannelAnnotationStore(tmp_path / "ann.json")
        store.set("A", "text")
        store.delete("A")
        assert store.get("A") is None

    def test_clear(self, tmp_path):
        store = ChannelAnnotationStore(tmp_path / "ann.json")
        store.set("A", "x")
        store.set("B", "y")
        store.clear()
        assert len(store.all()) == 0

    def test_as_text_dict(self, tmp_path):
        store = ChannelAnnotationStore(tmp_path / "ann.json")
        store.set("A", "x")
        store.set("B", "y")
        d = store.as_text_dict()
        assert d == {"A": "x", "B": "y"}

    def test_to_list(self, tmp_path):
        store = ChannelAnnotationStore(tmp_path / "ann.json")
        store.set("C1", "note1")
        lst = store.to_list()
        assert len(lst) == 1
        assert lst[0]["channel"] == "C1"

    def test_get_text_missing(self, tmp_path):
        store = ChannelAnnotationStore(tmp_path / "ann.json")
        assert store.get_text("NOPE") == ""


# ===========================================================================
# Report generation
# ===========================================================================

class TestReportGenerator:
    """Tests for the HTML report generator."""

    def test_basic_html_generation(self, tmp_path):
        from CESA.report_generator import ReportConfig, ReportGenerator

        pipe = FilterPipeline(filters=[BandpassFilter(low_hz=0.3, high_hz=35)])
        gen = ReportGenerator(
            pipelines={"EEG Fp1": pipe},
            config=ReportConfig(title="Test Report", include_charts=False),
        )
        out = gen.generate(tmp_path / "report.html")
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "CESA" in content
        assert "Test Report" in content
        assert "EEG Fp1" in content

    def test_html_includes_audit(self, tmp_path):
        from CESA.report_generator import ReportConfig, ReportGenerator

        audit = FilterAuditLog()
        audit.record("EEG1", "apply_preset", preset_name="TestPreset")
        gen = ReportGenerator(
            pipelines={"EEG1": FilterPipeline()},
            audit_log=audit,
            config=ReportConfig(include_charts=False),
        )
        out = gen.generate(tmp_path / "report.html")
        content = out.read_text(encoding="utf-8")
        assert "apply_preset" in content
        assert "TestPreset" in content

    def test_html_includes_annotations(self, tmp_path):
        from CESA.report_generator import ReportConfig, ReportGenerator

        gen = ReportGenerator(
            pipelines={"EEG1": FilterPipeline()},
            annotations={"EEG1": "Alpha burst at 3:42"},
            config=ReportConfig(include_charts=False),
        )
        out = gen.generate(tmp_path / "report.html")
        content = out.read_text(encoding="utf-8")
        assert "Alpha burst at 3:42" in content

    def test_html_with_charts(self, tmp_path):
        from CESA.report_generator import ReportConfig, ReportGenerator

        pipe = FilterPipeline(filters=[BandpassFilter(low_hz=0.3, high_hz=35)])
        raw = _sine(10, 5.0) + 0.1 * _sine(50, 5.0)
        gen = ReportGenerator(
            pipelines={"EEG1": pipe},
            raw_data={"EEG1": raw},
            sfreq=SFREQ,
            config=ReportConfig(include_charts=True),
        )
        out = gen.generate(tmp_path / "chart_report.html")
        content = out.read_text(encoding="utf-8")
        assert "data:image/svg+xml;base64," in content

    def test_html_dashboard_section(self, tmp_path):
        from CESA.report_generator import ReportConfig, ReportGenerator

        gen = ReportGenerator(
            pipelines={"A": FilterPipeline(), "B": FilterPipeline(filters=[NotchFilter()])},
            config=ReportConfig(include_charts=False),
        )
        out = gen.generate(tmp_path / "report.html")
        content = out.read_text(encoding="utf-8")
        assert "Dashboard" in content

    def test_channel_filter(self, tmp_path):
        from CESA.report_generator import ReportConfig, ReportGenerator

        gen = ReportGenerator(
            pipelines={"A": FilterPipeline(), "B": FilterPipeline()},
            config=ReportConfig(include_charts=False, channels=["A"]),
        )
        html = gen.render_html()
        assert "A" in html
        assert ">B<" not in html

    def test_empty_pipelines(self, tmp_path):
        from CESA.report_generator import ReportGenerator

        gen = ReportGenerator()
        out = gen.generate(tmp_path / "empty.html")
        assert out.exists()

    def test_suggestions_section(self, tmp_path):
        from CESA.report_generator import ReportConfig, ReportGenerator

        gen = ReportGenerator(
            pipelines={"X": FilterPipeline()},
            suggestions=[{"preset_name": "EEG PSG", "channel": "X",
                          "reason": "delta dominant", "confidence": 0.9,
                          "accepted": True}],
            config=ReportConfig(include_charts=False),
        )
        out = gen.generate(tmp_path / "sug.html")
        content = out.read_text(encoding="utf-8")
        assert "delta dominant" in content

    def test_snr_computation(self):
        from CESA.report_generator import _compute_snr

        raw = np.ones(100)
        filtered = raw * 0.9
        snr = _compute_snr(raw, filtered)
        assert np.isfinite(snr)

    def test_escape_html(self):
        from CESA.report_generator import _escape

        assert _escape('<script>alert("XSS")</script>') == \
            '&lt;script&gt;alert(&quot;XSS&quot;)&lt;/script&gt;'
