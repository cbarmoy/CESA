"""Toolkit-independent application controller for CESA.

Holds all business state (loaded recording, filters, scoring, profiles,
navigation) so that the Qt main window can operate without any Tkinter
dependency.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import mne
except ImportError:
    mne = None  # type: ignore[assignment]

from core.telemetry import telemetry
from core.raw_loader import open_raw_file, SUPPORTED_RECORDING_EXTENSIONS
from CESA.profile_store import ProfileStore
from CESA.profile_schema import (
    DisplayProcessingProfile, LayoutConfig, ScalingConfig, SignalSection,
    SpectrogramConfig,
)
from CESA.filter_engine import FilterAuditLog, FilterPipeline, pipeline_from_legacy_params
from CESA.filters import (
    apply_filter as cesa_apply_filter,
    apply_baseline_correction as cesa_apply_baseline_correction,
    detect_signal_type as cesa_detect_signal_type,
    get_filter_presets as cesa_get_filter_presets,
)

logger = logging.getLogger(__name__)

Signal = Tuple[np.ndarray, float]


class AppController:
    """Business-logic controller, independent of any GUI toolkit.

    The Qt ``EEGViewerMainWindow`` delegates data operations here.
    """

    def __init__(self) -> None:
        # -- Recording data ------------------------------------------------
        self.raw: Optional[Any] = None  # mne.io.Raw
        self.derivations: Dict[str, np.ndarray] = {}
        self.selected_channels: List[str] = []
        self.psg_channels_used: List[str] = []

        # -- Temporal navigation -------------------------------------------
        self.current_time: float = 0.0
        self.duration: float = 30.0
        self.sfreq: float = 200.0
        self.total_duration_s: float = 0.0

        # -- Filters -------------------------------------------------------
        self.filter_enabled: bool = True
        self.filter_low: float = 0.5
        self.filter_high: float = 30.0
        self.filter_type: str = "butterworth"
        self.filter_order: int = 4
        self.filter_window: str = "hamming"
        self.baseline_correction_enabled: bool = True
        self.baseline_window_duration: float = 30.0
        self.scaling: ScalingConfig = ScalingConfig()
        self.layout: LayoutConfig = LayoutConfig()
        self.spectrogram: SpectrogramConfig = SpectrogramConfig()

        self.channel_filter_params: Dict[str, Dict[str, float]] = {}
        self.channel_filter_pipelines: Dict[str, FilterPipeline] = {}
        self._filter_audit_log = FilterAuditLog()

        self.default_derivation_presets: Dict[str, Tuple[float, float]] = {
            'F3-M2': (0.3, 35.0), 'F4-M1': (0.3, 35.0),
            'C3-M2': (0.3, 35.0), 'C4-M1': (0.3, 35.0),
            'O1-M2': (0.3, 35.0), 'O2-M1': (0.3, 35.0),
            'F3': (0.3, 35.0), 'F4': (0.3, 35.0),
            'C3': (0.3, 35.0), 'C4': (0.3, 35.0),
            'O1': (0.3, 35.0), 'O2': (0.3, 35.0),
            'ECG': (0.3, 70.0), 'ECG1': (0.3, 70.0), 'ECG2': (0.3, 70.0),
            'EKG': (0.3, 70.0),
            'EMG': (10.0, 0.0), 'EMG Chin': (10.0, 0.0),
            'Chin': (10.0, 0.0), 'MENTON': (10.0, 0.0),
            'E1-M2': (0.3, 35.0), 'E2-M1': (0.3, 35.0),
            'EOG': (0.3, 35.0), 'EOG gauche': (0.3, 35.0),
            'EOG droite': (0.3, 35.0), 'EOG Left': (0.3, 35.0),
            'EOG Right': (0.3, 35.0),
        }

        # -- Scoring -------------------------------------------------------
        #: Scoring actif : celui affiché (hypnogramme, dashboard) et exporté.
        #: C'est une *copie* de l'un des deux stockages ci-dessous, jamais la
        #: référence utilisée pour la comparaison auto/manuel.
        self.sleep_scoring_data: Optional[pd.DataFrame] = None
        #: Résultat du dernier scoring automatique, préservé même si un scoring
        #: manuel est importé ensuite (sinon la comparaison sortirait à 100 %).
        self.auto_scoring_data: Optional[pd.DataFrame] = None
        #: Scoring humain de référence (import externe ou saisie dans l'éditeur).
        self.manual_scoring_data: Optional[pd.DataFrame] = None
        self.auto_scoring_method: Optional[str] = None
        self.auto_scoring_epoch_seconds: Optional[float] = None
        #: Settings and resolved channels of the last automatic run, reported
        #: with the comparison so a weak agreement can be traced to its cause.
        self.auto_scoring_params: Dict[str, Any] = {}
        self.manual_scoring_epoch_seconds: Optional[float] = None
        self.manual_scoring_source: Optional[str] = None
        self.active_scoring_source: Optional[str] = None
        self.sleep_scoring_method: str = "pftsleep"
        self.auto_scoring_epoch_length: float = 30.0
        self.scoring_epoch_duration: float = 30.0
        self.scoring_dirty: bool = False
        self.show_manual_scoring: bool = True
        #: Après un stade (toolbar / touches 0–5), avancer d'une époque dans le viewer Qt.
        self.manual_scoring_auto_advance: bool = True
        #: Dernier message d'erreur du scoring automatique (affiché par l'UI).
        self.last_auto_scoring_error: Optional[str] = None

        self.night_start_min: Optional[float] = None
        self.night_end_min: Optional[float] = None
        self.night_start_s: Optional[float] = None
        self.night_end_s: Optional[float] = None

        # YASA-specific
        self.yasa_eeg_candidates: List[str] = [
            'C4-M1', 'F4-M1', 'C3-M2', 'F3-M2', 'O2-M1', 'O1-M2',
            'C4', 'F4', 'C3', 'F3', 'O2', 'O1',
        ]
        self.yasa_eog_candidates: List[str] = [
            'E2-M1', 'E1-M2', 'EOG', 'EOG gauche', 'EOG droite',
        ]
        self.yasa_emg_candidates: List[str] = [
            "1-2", "1-F", "2-F",
            "Chin1-Chin2", "Chin1-Chin3", "EMG Chin", "EMG Menton",
            "Chin", "MENTON", "1", "2", "F", "EMG",
        ]
        #: Per-backend channel customisation, keyed by method name. Absent
        #: entries fall back to that backend's documented priorities, which
        #: differ from one model to the next.
        self.scoring_channel_overrides: Dict[str, Dict[str, List[str]]] = {}
        self.yasa_target_sfreq: float = 100.0
        self.yasa_confidence_threshold: float = 0.80
        self.yasa_age: Optional[int] = None
        self.yasa_male: Optional[bool] = None
        #: After automatic scoring, promote N2 epochs that meet the AASM N3
        #: slow-wave criterion on a frontal derivation. Off by default so a
        #: comparison measures the backend itself.
        self.n3_aasm_correction: bool = False

        # U-Sleep specific. The published weights are not distributed, so the
        # official web service (token below) is the normal path; a checkpoint
        # only serves those who trained or obtained one.
        self.usleep_checkpoint_path: Optional[str] = None
        self.usleep_sfreq: float = 128.0
        self.usleep_device: Optional[str] = None
        self.usleep_use_eog: bool = True
        self.usleep_zscore: bool = True
        #: Never persisted: a token expires after 12 h and identifies the user.
        self.usleep_api_token: Optional[str] = None

        # PFTSleep specific
        self.pft_models_dir: Optional[str] = None
        self.pft_device: str = "auto"
        self.pft_hf_token: Optional[str] = None
        self.pft_eeg_channel: Optional[str] = None
        self.pft_eog_channel: Optional[str] = None
        self.pft_emg_channel: Optional[str] = None
        self.pft_ecg_channel: Optional[str] = None

        # Pipeline ML specific
        self.ml_model_path: Optional[str] = None

        # -- Profiles ------------------------------------------------------
        self.profile_store = ProfileStore()
        self.active_profile: DisplayProcessingProfile = self.profile_store.load_last_or_default()
        self.profile_channel_map_runtime: Dict[str, str] = {}

        self._apply_profile_settings()

        # -- Data bridge (precomputed mode) --------------------------------
        self.data_bridge: Optional[Any] = None
        self.data_mode: str = "raw"

        # -- Thread pool ---------------------------------------------------
        self._preprocess_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=3)
        self._bridge_executor: Optional[ThreadPoolExecutor] = None
        self._processed_lock = threading.Lock()

        # -- Stage / display mapping ---------------------------------------
        self.french_to_standard: Dict[str, str] = {
            'eveil': 'W', 'w': 'W', 'wake': 'W', 'veille': 'W',
            'n1': 'N1', 'n2': 'N2', 'n3': 'N3', 'sws': 'N3',
            'rem': 'R', 'paradoxal': 'R', 'r': 'R',
            'mt': 'U', 'mvt': 'U', 'unk': 'U', 'inconnu': 'U',
        }
        self.sleep_stages: Dict[str, str] = {
            'W': 'Eveil', 'N1': 'N1', 'N2': 'N2', 'N3': 'N3',
            'R': 'REM', 'U': 'Inconnu',
        }

        # -- Misc ----------------------------------------------------------
        self.temporal_markers: List[Any] = []
        self.dark_theme_enabled: bool = True
        self.edf_path: Optional[str] = None

        # In-memory ocular cleaning backup (never written to disk)
        self._ocular_backup: Optional[Dict[str, np.ndarray]] = None
        self._ica_backup: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------
    # Profile helpers
    # ------------------------------------------------------------------

    def _apply_profile_settings(self) -> None:
        """Apply persisted profile settings on startup."""
        try:
            p = self.active_profile
            self.baseline_correction_enabled = bool(p.baseline_enabled)
            self.baseline_window_duration = float(p.baseline_window_duration)
            self.filter_enabled = bool(p.filter_enabled)
            self.filter_order = int(p.filter_order)
            if p.filter_low is not None:
                self.filter_low = float(p.filter_low)
            if p.filter_high is not None:
                self.filter_high = float(p.filter_high)
            self.filter_type = str(p.filter_type or "butterworth")
            self.filter_window = str(p.filter_window or "hamming")
            for ch, cfg in (p.channel_filter_params or {}).items():
                self.channel_filter_params[ch] = dict(cfg)
            for ch, pipe_dict in (p.channel_filter_pipelines or {}).items():
                try:
                    self.channel_filter_pipelines[ch] = FilterPipeline.from_dict(pipe_dict)
                except Exception:
                    pass
            self.scaling = ScalingConfig.from_dict(
                p.scaling.to_dict() if isinstance(p.scaling, ScalingConfig) else {}
            )
            self.layout = LayoutConfig.from_dict(
                p.layout.to_dict() if isinstance(p.layout, LayoutConfig) else {}
            )
            self.spectrogram = SpectrogramConfig.from_dict(
                p.spectrogram.to_dict() if isinstance(p.spectrogram, SpectrogramConfig) else {}
            )
        except Exception:
            pass

    def save_active_profile(self) -> None:
        p = self.active_profile
        p.filter_enabled = self.filter_enabled
        p.filter_low = self.filter_low
        p.filter_high = self.filter_high
        p.filter_order = self.filter_order
        p.filter_type = self.filter_type
        p.filter_window = self.filter_window
        p.baseline_enabled = self.baseline_correction_enabled
        p.baseline_window_duration = self.baseline_window_duration
        p.channel_filter_pipelines = {
            ch: pipe.to_dict() for ch, pipe in self.channel_filter_pipelines.items()
        }
        p.scaling = ScalingConfig.from_dict(self.scaling.to_dict())
        p.layout = LayoutConfig.from_dict(self.layout.to_dict())
        p.spectrogram = SpectrogramConfig.from_dict(self.spectrogram.to_dict())
        self.profile_store.save_profile(p)
        self.profile_store.set_last_profile_name(p.name)

    def compute_auto_gains(
        self,
        channel_types: Dict[str, str],
        target_spacing: float = 0.0,
    ) -> Dict[str, float]:
        """Compute per-channel gains so that each channel fills ~40% of spacing.

        Uses the 98th percentile of absolute amplitude over the full
        recording for robust estimation.  Returns ``{channel: gain}``.
        """
        spacing = target_spacing or self.scaling.spacing_uv
        target_half = spacing * 0.4
        gains: Dict[str, float] = {}
        for ch_name, arr in self.derivations.items():
            if arr.size < 2:
                continue
            step = max(1, arr.size // 200_000)
            sub = arr[::step]
            p98 = float(np.percentile(np.abs(sub), 98))
            if p98 < 1e-9:
                continue
            g = target_half / p98
            gains[ch_name] = max(0.01, min(g, 1e6))
        return gains

    def compute_signal_medians(self) -> Dict[str, float]:
        """Compute per-channel median for DC-offset centering.

        Returns ``{channel: median_uv}`` using subsampled data for speed.
        """
        medians: Dict[str, float] = {}
        for ch_name, arr in self.derivations.items():
            if arr.size < 2:
                continue
            step = max(1, arr.size // 200_000)
            sub = arr[::step]
            medians[ch_name] = float(np.median(sub))
        return medians

    def compute_auto_layout(
        self,
        channel_types: Dict[str, str],
    ) -> Dict[str, Any]:
        """Analyze channels and compute optimal layout multipliers.

        Returns a dict with keys ``multipliers``, ``center``, ``context``
        that can be applied to a ``LayoutConfig``.
        """
        type_counts: Dict[str, int] = {}
        type_amplitudes: Dict[str, List[float]] = {}

        for ch_name, arr in self.derivations.items():
            if arr.size < 2:
                continue
            ch_type = channel_types.get(ch_name, "eeg")
            type_counts[ch_type] = type_counts.get(ch_type, 0) + 1
            step = max(1, arr.size // 200_000)
            sub = arr[::step]
            p95 = float(np.percentile(np.abs(sub), 95))
            type_amplitudes.setdefault(ch_type, []).append(p95)

        nb_eeg = type_counts.get("eeg", 0)
        nb_emg = type_counts.get("emg", 0)
        nb_ecg = type_counts.get("ecg", 0)

        if nb_eeg >= 4:
            context = "eeg_detailed"
        elif nb_emg >= 2:
            context = "muscle_focus"
        else:
            context = "balanced"

        # Lane heights only — ECG/EMG amplitude is handled by per_type_gains
        # (shrinking the ECG lane made high-amplitude traces overflow worse).
        if context == "eeg_detailed":
            base = {"eeg": 1.6, "eog": 1.2, "emg": 0.6, "ecg": 0.7}
        elif context == "muscle_focus":
            base = {"eeg": 1.2, "eog": 1.0, "emg": 0.8, "ecg": 0.7}
        else:
            base = {"eeg": 1.3, "eog": 1.0, "emg": 0.7, "ecg": 0.7}

        ref_amp = 50.0
        for sig_type, amps in type_amplitudes.items():
            if not amps:
                continue
            # Do not shrink ECG/EMG lanes based on raw amplitude.
            if sig_type in ("ecg", "emg"):
                continue
            mean_amp = sum(amps) / len(amps)
            if mean_amp < 1e-3:
                continue
            ratio = ref_amp / mean_amp
            ratio = max(0.3, min(ratio, 2.0))
            base[sig_type] = base.get(sig_type, 1.0) * ratio

        for k in base:
            floor = 0.5 if k in ("ecg", "emg") else 0.1
            base[k] = round(max(floor, min(base[k], 3.0)), 2)

        return {
            "multipliers": base,
            "center": True,
            "context": context,
        }

    # ------------------------------------------------------------------
    # Recording loading
    # ------------------------------------------------------------------

    def load_recording(
        self,
        file_path: str,
        *,
        mode: str = "raw",
        ms_path: Optional[str] = None,
        precompute_action: str = "existing",
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Load a recording file and prepare derivations.

        Returns True on success.  *progress_callback* receives status
        messages suitable for display in a progress dialog.
        """
        def _progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            logger.info("[LOAD] %s", msg)

        try:
            _progress("Lecture du fichier EDF...")
            self.raw = open_raw_file(file_path, preload=True, verbose=False)
            self.sfreq = float(self.raw.info['sfreq'])
            self.edf_path = file_path
            n_ch = len(self.raw.ch_names)
            logger.info("[LOAD] Loaded %d channels at %.1f Hz", n_ch, self.sfreq)

            _progress("Creation des derivations...")
            self._create_derivations()
            self._ocular_backup = None
            self._ica_backup = None

            self.total_duration_s = len(self.raw.times) / self.sfreq if self.sfreq > 0 else 0.0
            self.current_time = 0.0

            # Resolve data mode
            is_lazy = mode in {"raw", "lazy"}
            self.data_mode = "lazy" if is_lazy else mode

            if mode == "precomputed" and ms_path:
                _progress("Configuration mode pre-calcule...")
                self._setup_precomputed(ms_path, precompute_action, _progress)

            _progress("Pret.")
            return True

        except Exception as exc:
            logger.error("Failed to load recording: %s", exc, exc_info=True)
            return False

    def _create_derivations(self) -> None:
        """Extract channel data arrays from the loaded raw recording.

        MNE stores electrical channels in Volts (SI).  The Qt viewer
        expects micro-volts so that the default 150 µV spacing produces
        readable traces.  We convert V -> µV here once, at load time.
        """
        if self.raw is None:
            return
        self.derivations.clear()
        data = self.raw.get_data()  # shape (n_channels, n_samples), Volts
        for idx, ch_name in enumerate(self.raw.ch_names):
            arr = np.asarray(data[idx], dtype=np.float64) * 1e6  # V -> µV
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            self.derivations[ch_name] = arr

    def _setup_precomputed(
        self, ms_path: str, action: str, progress: Callable[[str], None],
    ) -> None:
        """Set up the precomputed data bridge if possible."""
        from core.providers import PrecomputedProvider
        from core.bridge import DataBridge

        ms = Path(ms_path)
        valid = ms.exists() and (ms / ".zattrs").exists() and (ms / "levels").exists()

        if action == "existing" and not valid:
            logger.warning("[LOAD] Zarr directory invalid, falling back to lazy mode")
            self.data_mode = "lazy"
            return

        if action == "build":
            progress("Construction de la pyramide multiscale...")
            try:
                from core.multiscale import build_multiscale
                build_multiscale(self.raw, str(ms))
            except Exception as exc:
                logger.error("Multiscale build failed: %s", exc)
                self.data_mode = "lazy"
                return

        try:
            provider = PrecomputedProvider(ms)
            self._bridge_executor = self._bridge_executor or ThreadPoolExecutor(max_workers=2)
            self.data_bridge = DataBridge(provider, self._bridge_executor)
            self.data_mode = "precomputed"
        except Exception as exc:
            logger.error("PrecomputedProvider failed: %s", exc)
            self.data_mode = "lazy"

    # ------------------------------------------------------------------
    # Channel mapping for profile sections
    # ------------------------------------------------------------------

    def auto_map_channels(self, channels: List[str]) -> Dict[str, str]:
        """Return best-guess channel-to-section mapping."""
        mapping: Dict[str, str] = {}
        for ch in channels:
            sig_type = cesa_detect_signal_type(ch)
            mapping[ch] = sig_type
        return mapping

    def get_ordered_channels_for_viewer(self) -> List[str]:
        """Return channels ordered by profile section."""
        if not self.raw:
            return []
        section_order = [
            s.key for s in self.active_profile.signal_sections if bool(s.enabled)
        ]
        ordered: List[str] = []
        for section_key in section_order:
            section_chs = [
                ch for ch, mapped in self.profile_channel_map_runtime.items()
                if mapped == section_key and ch in self.raw.ch_names
            ]
            ordered.extend(section_chs)
        return ordered or list(self.raw.ch_names)

    # ------------------------------------------------------------------
    # Signals for the viewer
    # ------------------------------------------------------------------

    def build_viewer_signals(
        self,
        channels: Optional[List[str]] = None,
    ) -> Dict[str, Signal]:
        """Build ``{name: (data, sfreq)}`` dict for the Qt viewer."""
        if self.raw is None:
            return {}
        chs = channels or self.psg_channels_used or list(self.raw.ch_names)
        signals: Dict[str, Signal] = {}
        for ch in chs:
            if ch in self.derivations:
                signals[ch] = (self.derivations[ch], self.sfreq)
        return signals

    def build_viewer_filter_pipelines(self) -> Dict[str, FilterPipeline]:
        return dict(self.channel_filter_pipelines)

    # ------------------------------------------------------------------
    # Ocular cleaning (in-memory only; never writes recording files)
    # ------------------------------------------------------------------

    def has_ocular_cleaning(self) -> bool:
        """True if an in-memory ocular cleaning backup is active."""
        return bool(self._ocular_backup)

    def apply_ocular_cleaning(
        self,
        cleaned: Dict[str, np.ndarray],
        *,
        originals: Optional[Dict[str, np.ndarray]] = None,
    ) -> int:
        """Replace EEG arrays with cleaned versions (memory only).

        Backs up each channel the first time it is cleaned so
        :meth:`restore_ocular_backup` can undo. Updates ``derivations``
        when the channel name matches; otherwise only the backup is kept
        for viewer-level restore. Does not touch ``raw`` or write files.

        Parameters
        ----------
        cleaned :
            ``{channel: cleaned_array}``
        originals :
            Optional ``{channel: original_array}`` used for backup when the
            channel is not present in ``derivations`` (e.g. renamed display
            names). Ignored for channels already backed up.

        Returns
        -------
        int
            Number of channels cleaned (backup entries created or updated).
        """
        if not cleaned:
            return 0
        if self._ocular_backup is None:
            self._ocular_backup = {}
        originals = originals or {}
        n = 0
        for name, arr in cleaned.items():
            if name not in self._ocular_backup:
                if name in self.derivations:
                    self._ocular_backup[name] = np.asarray(
                        self.derivations[name], dtype=float
                    ).copy()
                elif name in originals:
                    self._ocular_backup[name] = np.asarray(
                        originals[name], dtype=float
                    ).copy()
                else:
                    continue
            if name in self.derivations:
                self.derivations[name] = np.asarray(arr, dtype=float)
            n += 1
        logger.info(
            "[OCULAR] Applied in-memory cleaning to %d channel(s) (file unchanged)",
            n,
        )
        return n

    def restore_ocular_backup(self) -> Dict[str, np.ndarray]:
        """Restore EEG arrays from the ocular cleaning backup.

        Writes matching keys back into ``derivations`` and returns a copy
        of all restored arrays so the viewer can refresh display names.
        """
        if not self._ocular_backup:
            return {}
        restored: Dict[str, np.ndarray] = {}
        for name, arr in self._ocular_backup.items():
            copy = np.asarray(arr, dtype=float).copy()
            restored[name] = copy
            if name in self.derivations:
                self.derivations[name] = copy
        self._ocular_backup = None
        logger.info("[OCULAR] Restored %d channel(s) from backup", len(restored))
        return restored

    def has_ica_cleaning(self) -> bool:
        """True if an in-memory ICA cleaning backup is active."""
        return bool(self._ica_backup)

    def apply_ica_cleaning(
        self,
        cleaned: Dict[str, np.ndarray],
        *,
        originals: Optional[Dict[str, np.ndarray]] = None,
    ) -> int:
        """Replace EEG arrays with ICA-cleaned versions (memory only)."""
        if not cleaned:
            return 0
        if self._ica_backup is None:
            self._ica_backup = {}
        originals = originals or {}
        n = 0
        for name, arr in cleaned.items():
            if name not in self._ica_backup:
                if name in self.derivations:
                    self._ica_backup[name] = np.asarray(
                        self.derivations[name], dtype=float
                    ).copy()
                elif name in originals:
                    self._ica_backup[name] = np.asarray(
                        originals[name], dtype=float
                    ).copy()
                else:
                    continue
            if name in self.derivations:
                self.derivations[name] = np.asarray(arr, dtype=float)
            n += 1
        logger.info(
            "[ICA] Applied in-memory cleaning to %d channel(s) (file unchanged)",
            n,
        )
        return n

    def restore_ica_backup(self) -> Dict[str, np.ndarray]:
        """Restore EEG arrays from the ICA cleaning backup."""
        if not self._ica_backup:
            return {}
        restored: Dict[str, np.ndarray] = {}
        for name, arr in self._ica_backup.items():
            copy = np.asarray(arr, dtype=float).copy()
            restored[name] = copy
            if name in self.derivations:
                self.derivations[name] = copy
        self._ica_backup = None
        logger.info("[ICA] Restored %d channel(s) from backup", len(restored))
        return restored

    def has_embedded_scoring(self) -> bool:
        """Check if the loaded EDF+ file contains sleep stage annotations."""
        if self.raw is None:
            return False
        try:
            ann = self.raw.annotations
            if ann is None or len(ann) == 0:
                return False
            return any("Sleep stage" in str(d) for d in ann.description)
        except Exception:
            return False

    def import_embedded_scoring(self) -> bool:
        """Extract sleep scoring from the loaded EDF+'s own annotations."""
        if self.raw is None:
            return False
        try:
            from CESA.scoring_io import import_edf_hypnogram
            abs_start = None
            if self.raw.info.get("meas_date"):
                abs_start = pd.Timestamp(self.raw.info["meas_date"])
            df = import_edf_hypnogram(
                self.edf_path,
                recording_duration_s=self.total_duration_s,
                epoch_seconds=self.scoring_epoch_duration,
                absolute_start_datetime=abs_start,
            )
            if df is not None and not df.empty:
                self.manual_scoring_data = df.reset_index(drop=True).copy()
                self.manual_scoring_epoch_seconds = float(self.scoring_epoch_duration)
                self.manual_scoring_source = "annotations EDF+ integrees"
                self.sleep_scoring_data = df
                self.active_scoring_source = "manual"
                self.scoring_dirty = False
                logger.info("[SCORING] Imported embedded scoring: %d epochs", len(df))
                return True
        except Exception as exc:
            logger.error("Embedded scoring import failed: %s", exc, exc_info=True)
        return False

    def ensure_scoring_dataframe(self) -> bool:
        """Create a blank scoring DataFrame (all 'U') if none exists yet.

        Returns True if a DataFrame is available after the call.
        """
        if self.sleep_scoring_data is not None and not self.sleep_scoring_data.empty:
            return True
        if self.raw is None or self.total_duration_s <= 0:
            return False
        epoch_len = self.scoring_epoch_duration
        n_epochs = int(np.ceil(self.total_duration_s / epoch_len))
        times = np.arange(n_epochs) * epoch_len
        self.sleep_scoring_data = pd.DataFrame({
            "time": times,
            "stage": ["U"] * n_epochs,
        })
        self.scoring_dirty = True
        logger.info("[SCORING] Created blank scoring: %d epochs of %.0f s", n_epochs, epoch_len)
        return True

    def build_hypnogram_tuple(self) -> Optional[Tuple[List[str], float]]:
        """Return ``(labels, epoch_len)`` or None."""
        df = self.sleep_scoring_data
        if df is None or df.empty:
            return None
        labels = list(df['stage'].astype(str))
        return (labels, self.scoring_epoch_duration)

    def build_scoring_annotations(self) -> List[Dict[str, Any]]:
        """Build event list for the events bar."""
        df = self.sleep_scoring_data
        if df is None or df.empty:
            return []
        events: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            events.append({
                "onset": float(row.get("time", 0.0)),
                "duration": float(self.scoring_epoch_duration),
                "description": str(row.get("stage", "U")),
            })
        return events

    def channel_types_dict(self) -> Dict[str, str]:
        """Return ``{channel: type}`` for all loaded channels."""
        if self.raw is None:
            return {}
        return {ch: cesa_detect_signal_type(ch) for ch in self.raw.ch_names}

    @staticmethod
    def resolve_default_spectrogram_channel(available: Sequence[str]) -> str:
        """Pick a sensible default channel for the wavelet spectrogram bars.

        Priority: exact ``D11`` (case-insensitive) > a C3 derivation
        referenced to M1/M2/A1/A2 (e.g. ``C3-M1``) > first channel detected
        as EEG > first available channel.
        """
        names = list(available)
        if not names:
            return ""
        for n in names:
            if n.strip().lower() == "d11":
                return n
        for n in names:
            low = n.lower()
            if "c3" in low and any(ref in low for ref in ("m1", "m2", "a1", "a2")):
                return n
        for n in names:
            if cesa_detect_signal_type(n) == "eeg":
                return n
        return names[0]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    SUPPORTED_SCORING_METHODS = (
        "pftsleep", "yasa", "usleep",
        "aasm_rules", "ml", "ml_hmm", "rules_hmm",
    )

    def auto_scoring_preflight(self, method: Optional[str] = None) -> List[str]:
        """Return blocking problems preventing *method* from running.

        An empty list means the backend can be launched. Checks the presence of
        the Python package, of the model weights and of a usable EEG channel so
        the UI can explain the failure instead of a bare "scoring echoue".
        """
        selected = (method or self.sleep_scoring_method or "").lower().strip()
        problems: List[str] = []

        if self.raw is None:
            return ["Aucun enregistrement charge."]

        available = {ch.upper().strip() for ch in self.raw.ch_names}
        if selected in ("yasa", "usleep", "aasm_rules", "ml", "ml_hmm", "rules_hmm"):
            eeg_candidates = self.effective_channel_candidates(selected)["eeg"]
            if not any(c.upper().strip() in available for c in eeg_candidates):
                problems.append(
                    "Aucun canal EEG candidat n'existe dans l'enregistrement : "
                    "selectionnez au moins une derivation EEG presente."
                )

        if selected == "yasa":
            if importlib.util.find_spec("yasa") is None:
                problems.append("Le paquet 'yasa' n'est pas installe (pip install yasa).")
        elif selected == "usleep":
            problems.extend(self._usleep_preflight())
        elif selected == "pftsleep":
            if importlib.util.find_spec("torch") is None:
                problems.append("PyTorch n'est pas installe (requis par PFTSleep).")
            if importlib.util.find_spec("pftsleep") is None:
                problems.append(
                    "Le paquet 'PFTSleep' n'est pas installe : impossible de charger le modele."
                )
            if not self.edf_path or not os.path.isfile(str(self.edf_path)):
                problems.append(
                    "PFTSleep relit le fichier EDF d'origine : chargez un EDF depuis le disque."
                )
        elif selected in ("ml", "ml_hmm"):
            model = (self.ml_model_path or "").strip()
            if not model:
                problems.append(
                    "Les backends ML necessitent un modele entraine : renseignez "
                    "'Modele ML' (aucun modele n'est fourni avec CESA)."
                )
            elif not os.path.isfile(model):
                problems.append(f"Modele ML introuvable : {model}")

        return problems

    def usleep_token(self) -> Optional[str]:
        """API token entered in the dialog, else ``USLEEP_API_TOKEN``."""
        token = str(self.usleep_api_token or "").strip()
        if token:
            return token
        return str(os.environ.get("USLEEP_API_TOKEN", "") or "").strip() or None

    def usleep_uses_api(self) -> bool:
        """True when the run will go through the official web service."""
        ckpt = (self.usleep_checkpoint_path or "").strip()
        if self.usleep_token():
            return True
        return not (ckpt and os.path.isfile(ckpt))

    def _usleep_preflight(self) -> List[str]:
        """Blocking problems for U-Sleep, web service or local checkpoint."""
        problems: List[str] = []
        ckpt = (self.usleep_checkpoint_path or "").strip()

        if not self.usleep_uses_api():
            if not os.path.isfile(ckpt):
                problems.append(f"Checkpoint U-Sleep introuvable : {ckpt}")
            if importlib.util.find_spec("torch") is None:
                problems.append("PyTorch n'est pas installe (requis par U-Sleep).")
            if importlib.util.find_spec("braindecode") is None:
                problems.append(
                    "Le paquet 'braindecode' n'est pas installe (requis par U-Sleep)."
                )
            return problems

        if not self.usleep_token():
            problems.append(
                "U-Sleep passe par le service officiel : renseignez un token API "
                "(https://sleep.ai.ku.dk, Account -> Generate API Token, valable "
                "12 h) ou definissez USLEEP_API_TOKEN. A defaut, fournissez un "
                "checkpoint local."
            )
        if importlib.util.find_spec("usleep_api") is None:
            problems.append(
                "Le paquet 'usleep-api' n'est pas installe (pip install usleep-api)."
            )
        # Without an EDF writer only the file on disk can be uploaded.
        if importlib.util.find_spec("edfio") is None and (
            not self.edf_path or not os.path.isfile(str(self.edf_path))
        ):
            problems.append(
                "L'envoi au service exige un fichier : installez 'edfio' "
                "(pip install edfio) pour n'envoyer que les canaux de scoring, "
                "ou chargez un EDF depuis le disque."
            )
        return problems

    def channel_candidates_for(
        self, method: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """Channel candidates to use for *method*.

        The user's own selection for that backend wins; otherwise the backend's
        documented priorities apply. Modalities a backend ignores come back
        empty, so U-Sleep is never handed an EMG.
        """
        from CESA.scoring_channels import channel_profile

        selected = (method or self.sleep_scoring_method or "").lower().strip()
        profile = channel_profile(selected)
        override = self.scoring_channel_overrides.get(selected, {})
        out: Dict[str, List[str]] = {}
        for modality in ("eeg", "eog", "emg"):
            if not profile.uses(modality):
                out[modality] = []
                continue
            custom = override.get(modality)
            out[modality] = (
                list(custom) if custom is not None
                else list(profile.candidates(modality))
            )
        return out

    def set_channel_candidates_for(
        self,
        method: Optional[str],
        eeg: Optional[Sequence[str]] = None,
        eog: Optional[Sequence[str]] = None,
        emg: Optional[Sequence[str]] = None,
    ) -> None:
        """Remember the user's channel choice for *method* and activate it."""
        selected = (method or self.sleep_scoring_method or "").lower().strip()
        override = dict(self.scoring_channel_overrides.get(selected, {}))
        for modality, value in (("eeg", eeg), ("eog", eog), ("emg", emg)):
            if value is not None:
                override[modality] = list(value)
        self.scoring_channel_overrides[selected] = override
        self.apply_channel_profile(selected)

    def apply_channel_profile(self, method: Optional[str] = None) -> None:
        """Load *method*'s candidates into the active scoring channel lists."""
        candidates = self.channel_candidates_for(method)
        self.yasa_eeg_candidates = candidates["eeg"]
        self.yasa_eog_candidates = candidates["eog"]
        self.yasa_emg_candidates = candidates["emg"]

    def effective_channel_candidates(
        self, method: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """Candidates that will actually be handed to *method*.

        The active lists win, so setting them directly keeps working, and a
        modality the backend ignores is dropped. An empty list is taken
        literally as "no channel for this modality": substituting a default
        there would make it impossible to score deliberately without an EOG or
        an EMG, and would silently invalidate any measurement of their effect.
        """
        from CESA.scoring_channels import channel_profile

        selected = (method or self.sleep_scoring_method or "").lower().strip()
        profile = channel_profile(selected)
        active = {
            "eeg": self.yasa_eeg_candidates,
            "eog": self.yasa_eog_candidates,
            "emg": self.yasa_emg_candidates,
        }
        return {
            modality: list(active[modality]) if profile.uses(modality) else []
            for modality in ("eeg", "eog", "emg")
        }

    def resolved_scoring_channel(self, candidates: Sequence[str]) -> Optional[str]:
        """First candidate actually present in the recording, as the scorer does."""
        if self.raw is None:
            return None
        by_upper = {ch.upper().strip(): ch for ch in self.raw.ch_names}
        for candidate in candidates:
            name = str(candidate).strip()
            if not name:
                continue
            resolved = by_upper.get(name.upper())
            if resolved is not None:
                return resolved
        return None

    def auto_scoring_warnings(self, method: Optional[str] = None) -> List[str]:
        """Non-blocking issues likely to degrade the automatic scoring.

        Unlike :meth:`auto_scoring_preflight`, these do not prevent the run: they
        warn about a configuration that will produce a usable but poorer result.
        """
        from CESA.scoring_channels import emg_warning, find_submental_emg_candidates

        selected = (method or self.sleep_scoring_method or "").lower().strip()
        warnings: List[str] = []
        if self.raw is None or selected not in ("yasa", "usleep", "pftsleep"):
            return warnings

        emg = self.resolved_scoring_channel(
            self.effective_channel_candidates(selected)["emg"]
        )
        issue = emg_warning(emg)
        if issue:
            warnings.append(issue)
        better = find_submental_emg_candidates(self.raw.ch_names)
        if issue and better:
            warnings.append(
                "Canal mentonnier disponible dans l'enregistrement : "
                + ", ".join(better)
            )
        elif not emg and better and selected != "usleep":
            warnings.append(
                "Un EMG mentonnier est present dans l'enregistrement ("
                + ", ".join(better)
                + ") mais n'est pas utilise. Sur beaucoup de PSG francaises "
                "les trois electrodes du menton s'appellent 1, 2 et F : "
                "selectionnez la derivation bipolaire 1-2 si elle existe, "
                "sinon 1."
            )
        return warnings

    def run_auto_scoring(
        self,
        method: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Run automatic sleep scoring. Returns True on success.

        The result is kept in :attr:`auto_scoring_data` *and* published as the
        active scoring. :attr:`manual_scoring_data` is never touched, so a
        manual scoring imported afterwards can still be compared against it.
        On failure, :attr:`last_auto_scoring_error` holds the reason.
        """
        self.last_auto_scoring_error = None
        if self.raw is None:
            self.last_auto_scoring_error = "Aucun enregistrement charge."
            return False
        selected = (method or self.sleep_scoring_method).lower().strip()
        if selected not in self.SUPPORTED_SCORING_METHODS:
            selected = "pftsleep"
        self.sleep_scoring_method = selected

        if progress_callback:
            progress_callback(f"Scoring automatique ({selected})...")

        # Each backend has its own channel expectations; a modality it ignores
        # must not be passed at all (U-Sleep and the EMG, for instance).
        candidates = self.effective_channel_candidates(selected)

        # The backend reports the channels it actually resolved; capturing them
        # is the only way the report can say what was really fed to the model.
        used: Dict[str, Any] = {}

        def _on_event(stage: str, payload: Dict[str, Any]) -> None:
            for key in ("eeg", "eog", "emg", "ecg", "device", "source"):
                value = payload.get(key)
                if value:
                    used[key] = value
            if stage == "staging_initialized" and payload.get("metadata") is not None:
                used["metadata"] = bool(payload["metadata"])
            if stage == "resample_end" and payload.get("target_sfreq"):
                used["effective_sfreq"] = float(payload["target_sfreq"])

        started = time.time()
        try:
            from CESA.sleep_scorer import SleepScorer
            scorer = SleepScorer(
                method=selected,
                progress_cb=_on_event,
                epoch_length=self.auto_scoring_epoch_length,
                eeg_candidates=tuple(candidates["eeg"]),
                eog_candidates=tuple(candidates["eog"]),
                emg_candidates=tuple(candidates["emg"]),
                target_sfreq=self.yasa_target_sfreq,
                yasa_confidence_threshold=self.yasa_confidence_threshold,
                yasa_age=self.yasa_age,
                yasa_male=self.yasa_male,
                edf_path=self.edf_path,
                usleep_checkpoint_path=self.usleep_checkpoint_path,
                usleep_sfreq=self.usleep_sfreq,
                usleep_device=self.usleep_device,
                usleep_use_eog=self.usleep_use_eog,
                usleep_zscore=self.usleep_zscore,
                usleep_api_token=self.usleep_token(),
                pft_models_dir=self.pft_models_dir,
                pft_device=self.pft_device,
                pft_hf_token=self.pft_hf_token,
                pft_eeg_channel=self.pft_eeg_channel,
                pft_eog_channel=self.pft_eog_channel,
                pft_emg_channel=self.pft_emg_channel,
                pft_ecg_channel=self.pft_ecg_channel,
                ml_model_path=self.ml_model_path,
            )
            df = scorer.score(self.raw)
            if df is not None and not df.empty:
                epoch_len = float(self.auto_scoring_epoch_length)
                if self.n3_aasm_correction:
                    df, n_promoted, n3_channel = self._apply_n3_correction(
                        df, used.get("eeg"),
                    )
                    used["n3_correction"] = n_promoted
                    used["canal_n3"] = n3_channel
                self.auto_scoring_data = df
                self.auto_scoring_method = selected
                self.auto_scoring_epoch_seconds = epoch_len
                self.auto_scoring_params = self._build_auto_scoring_params(
                    selected, df, used, time.time() - started, candidates,
                )
                self.sleep_scoring_data = df.copy()
                self.active_scoring_source = "auto"
                self.scoring_epoch_duration = epoch_len
                self.scoring_dirty = True
                logger.info("[SCORING] %s: %d epochs", selected, len(df))
                return True
            self.last_auto_scoring_error = "Le backend n'a produit aucune epoque."
        except Exception as exc:
            self.last_auto_scoring_error = str(exc)
            logger.error("Auto scoring failed: %s", exc, exc_info=True)
        return False

    def _eeg_signal_stats(self, channel: Optional[str]) -> Dict[str, Any]:
        """Amplitude of the EEG channel the backend actually used.

        Purely diagnostic: it tells a signal that carries no slow wave from a
        model that failed to report them, so a failure here is never fatal.
        """
        if self.raw is None or not channel:
            return {}
        try:
            from CESA.scoring_signal_check import eeg_signal_stats

            data = self.raw.get_data(picks=[channel])
            stats = eeg_signal_stats(
                data,
                float(self.raw.info["sfreq"]),
                epoch_length=float(self.auto_scoring_epoch_length),
            )
            if stats:
                stats["canal_mesure"] = channel
            return stats or {}
        except Exception as exc:
            logger.debug("[SCORING] EEG signal check failed: %s", exc)
            return {}

    def _apply_n3_correction(
        self,
        df: pd.DataFrame,
        used_eeg: Optional[str],
    ):
        """Promote N2 to N3 where the frontal EEG meets the AASM criterion."""
        from CESA.scoring_channels import pick_n3_channel
        from CESA.scoring_signal_check import n3_epoch_mask, promote_n3

        channel = pick_n3_channel(list(self.raw.ch_names), fallback=used_eeg)
        if not channel:
            return df, 0, None
        try:
            data = self.raw.get_data(picks=[channel])
            mask = n3_epoch_mask(
                data,
                float(self.raw.info["sfreq"]),
                epoch_length=float(self.auto_scoring_epoch_length),
            )
        except Exception as exc:
            logger.debug("[SCORING] N3 correction failed: %s", exc)
            return df, 0, channel
        if mask is None:
            return df, 0, channel
        stages, promoted = promote_n3(list(df["stage"]), mask)
        out = df.copy()
        out["stage"] = stages
        return out, promoted, channel

    def _build_auto_scoring_params(
        self,
        method: str,
        df: pd.DataFrame,
        used: Dict[str, Any],
        elapsed_s: float,
        candidates: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Snapshot of everything that shaped the automatic scoring result.

        Reported alongside the comparison so a disappointing agreement can be
        traced back to the settings and the channels actually used.
        """
        params: Dict[str, Any] = {
            "backend": method,
            "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duree_calcul_s": round(float(elapsed_s), 1),
            "epoque_s": float(self.auto_scoring_epoch_length),
            "n_epoques": int(len(df)),
            "canal_eeg": used.get("eeg"),
            "canal_eog": used.get("eog"),
            "canal_emg": used.get("emg"),
            "canal_ecg": used.get("ecg"),
            "candidats_eeg": candidates["eeg"],
            "candidats_eog": candidates["eog"],
            "candidats_emg": candidates["emg"],
            "correction_n3_aasm": bool(self.n3_aasm_correction),
        }
        if used.get("canal_n3"):
            params["canal_n3"] = used.get("canal_n3")
        if used.get("n3_correction") is not None:
            params["epoques_promues_n3"] = used.get("n3_correction")
        if self.raw is not None:
            try:
                params["sfreq_enregistrement_hz"] = float(self.raw.info["sfreq"])
                params["n_canaux_disponibles"] = len(self.raw.ch_names)
            except Exception:
                pass
        if used.get("effective_sfreq"):
            params["sfreq_effective_hz"] = float(used["effective_sfreq"])
        if used.get("device"):
            params["peripherique"] = str(used["device"])

        if method == "yasa":
            params["sfreq_cible_hz"] = float(self.yasa_target_sfreq)
            params["seuil_confiance"] = float(self.yasa_confidence_threshold)
            params["age"] = self.yasa_age
            params["sexe"] = (
                None if self.yasa_male is None else ("homme" if self.yasa_male else "femme")
            )
        elif method == "usleep":
            via_api = str(used.get("source", "")) == "api"
            params["source_modele"] = (
                "service officiel sleep.ai.ku.dk" if via_api else "checkpoint local"
            )
            params["sfreq_cible_hz"] = float(self.usleep_sfreq)
            params["eog_en_entree"] = bool(self.usleep_use_eog)
            if via_api:
                params["modele_distant"] = "U-Sleep v1.0"
                params["envoi_anonymise"] = True
            else:
                params["checkpoint"] = self.usleep_checkpoint_path
                params["zscore_par_canal"] = bool(self.usleep_zscore)
        elif method == "pftsleep":
            params["dossier_poids"] = self.pft_models_dir or "PFTSleep/"
            params["peripherique"] = params.get("peripherique") or self.pft_device
        else:
            params["sfreq_cible_hz"] = float(self.yasa_target_sfreq)
            if method in ("ml", "ml_hmm"):
                params["modele_ml"] = self.ml_model_path

        stages = df["stage"].astype(str)
        total = len(stages) or 1
        params["repartition_stades"] = {
            stage: {
                "n": int(count),
                "pct": round(count / total * 100, 1),
            }
            for stage, count in stages.value_counts().items()
        }
        never = [s for s in ("W", "N1", "N2", "N3", "R") if s not in set(stages)]
        params["stades_jamais_predits"] = never
        signal = self._eeg_signal_stats(used.get("eeg"))
        if signal:
            params["signal_eeg"] = signal
        if "confidence" in df.columns:
            conf = pd.to_numeric(df["confidence"], errors="coerce").dropna()
            if len(conf):
                threshold = float(self.yasa_confidence_threshold)
                params["confiance_moyenne"] = round(float(conf.mean()), 3)
                params["confiance_mediane"] = round(float(conf.median()), 3)
                params["epoques_sous_seuil"] = int((conf < threshold).sum())
                params["pct_sous_seuil"] = round(
                    float((conf < threshold).mean() * 100), 1
                )
        return params

    def set_active_scoring(
        self,
        df: pd.DataFrame,
        source: str = "manual",
        *,
        epoch_seconds: Optional[float] = None,
        manual_source: Optional[str] = None,
    ) -> None:
        """Publish *df* as the active scoring and store it under *source*.

        *source* is ``"auto"`` or ``"manual"``; anything else only updates the
        active scoring without touching either reference store.
        """
        clean = df.reset_index(drop=True).copy()
        epoch_len = float(epoch_seconds) if epoch_seconds else float(self.scoring_epoch_duration)
        if source == "auto":
            self.auto_scoring_data = clean.copy()
            self.auto_scoring_epoch_seconds = epoch_len
        elif source == "manual":
            self.manual_scoring_data = clean.copy()
            self.manual_scoring_epoch_seconds = epoch_len
            if manual_source:
                self.manual_scoring_source = manual_source
            elif self.manual_scoring_source is None:
                self.manual_scoring_source = "editeur"
        self.sleep_scoring_data = clean
        self.scoring_epoch_duration = epoch_len
        if source in ("auto", "manual"):
            self.active_scoring_source = source
        self.scoring_dirty = True

    def copy_auto_scoring_to_manual(self) -> bool:
        """Seed the manual scoring with the automatic one as a starting point.

        Concordance stays at 100 % until epochs are actually edited, so callers
        must warn the user before offering this.
        """
        if self.auto_scoring_data is None or self.auto_scoring_data.empty:
            return False
        self.manual_scoring_data = self.auto_scoring_data.reset_index(drop=True).copy()
        self.manual_scoring_epoch_seconds = (
            self.auto_scoring_epoch_seconds or self.scoring_epoch_duration
        )
        self.manual_scoring_source = "copie du scoring auto"
        logger.info("[SCORING] Manual scoring seeded from auto (%d epochs)",
                    len(self.manual_scoring_data))
        return True

    def compare_scoring(self, exclude_unscored: bool = True) -> Optional[pd.DataFrame]:
        """Compare the automatic and the manual scoring, epoch by epoch.

        Returns a DataFrame with columns ``epoch``, ``auto``, ``manual``,
        ``match`` (labels normalised to W/N1/N2/N3/R/U), or None when one of the
        two scorings is missing. Diagnostics about the alignment live in
        ``result.attrs["diagnostics"]`` so the UI can flag a suspicious result,
        typically two hypnograms that turn out to be identical.
        """
        auto = self.auto_scoring_data
        manual = self.manual_scoring_data
        auto_is_fallback = False
        if (auto is None or auto.empty) and self.active_scoring_source != "manual":
            # Backward compatibility: scoring loaded before the stores were split.
            auto = self.sleep_scoring_data
            auto_is_fallback = auto is not None and not auto.empty
        if auto is None or manual is None or auto.empty or manual.empty:
            return None
        try:
            same_object = auto is manual
            auto_len = float(
                self.auto_scoring_epoch_seconds or self.scoring_epoch_duration or 30.0
            )
            manual_len = float(
                self.manual_scoring_epoch_seconds or self.scoring_epoch_duration or 30.0
            )
            # Align on the coarser grid so a 30 s scoring can be compared with a
            # 300 s one without silently mismatching epoch indices.
            common_len = max(auto_len, manual_len)

            auto_e = self._epoch_stage_frame(auto, auto_len, common_len, "auto")
            manual_e = self._epoch_stage_frame(manual, manual_len, common_len, "manual")
            if auto_e is None or manual_e is None:
                return None

            merged = pd.merge(auto_e, manual_e, on="epoch", how="inner")
            n_common_raw = len(merged)

            n_excluded_u = 0
            if exclude_unscored:
                keep = (merged["auto"] != "U") & (merged["manual"] != "U")
                n_excluded_u = int((~keep).sum())
                merged = merged[keep]

            merged = merged.reset_index(drop=True)
            merged["match"] = merged["auto"] == merged["manual"]

            n_before_window = len(merged)
            if self.has_night_window:
                start_s = float(self.night_start_s)  # type: ignore[arg-type]
                end_s = float(self.night_end_s)  # type: ignore[arg-type]
                epoch_start_s = merged["epoch"] * common_len
                merged = merged[
                    (epoch_start_s >= start_s) & (epoch_start_s <= end_s)
                ].reset_index(drop=True)

            identical = bool(
                len(auto_e) == len(manual_e)
                and n_common_raw == len(auto_e)
                and (
                    auto_e.sort_values("epoch")["auto"].to_numpy()
                    == manual_e.sort_values("epoch")["manual"].to_numpy()
                ).all()
            )

            merged.attrs["diagnostics"] = {
                "n_auto": int(len(auto_e)),
                "n_manual": int(len(manual_e)),
                "n_common": int(n_common_raw),
                "n_compared": int(len(merged)),
                "n_excluded_u": n_excluded_u,
                "n_excluded_night": int(n_before_window - len(merged)),
                "auto_epoch_seconds": auto_len,
                "manual_epoch_seconds": manual_len,
                "comparison_epoch_seconds": common_len,
                "auto_method": self.auto_scoring_method,
                "manual_source": self.manual_scoring_source,
                "auto_is_fallback": auto_is_fallback,
                "same_object": same_object,
                "identical": identical,
                "exclude_unscored": bool(exclude_unscored),
                "auto_params": dict(self.auto_scoring_params),
                # A stage the expert scored but the model never output at all
                # points to a signal problem rather than a borderline judgement.
                "never_predicted": [
                    stage
                    for stage in ("W", "N1", "N2", "N3", "R")
                    if stage in set(merged["manual"]) and stage not in set(merged["auto"])
                ],
            }
            return merged
        except Exception as exc:
            logger.error("Scoring comparison failed: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _epoch_stage_frame(
        df: pd.DataFrame,
        epoch_seconds: float,
        common_seconds: float,
        column: str,
    ) -> Optional[pd.DataFrame]:
        """Project a scoring DataFrame onto the ``common_seconds`` epoch grid.

        Labels are normalised, duplicated epochs are collapsed by majority vote
        (which also handles projecting a fine grid onto a coarser one).
        """
        from CESA.scoring_io import normalize_stage_series

        if df is None or df.empty or "stage" not in df.columns:
            return None
        out = pd.DataFrame({"stage": normalize_stage_series(df["stage"])})
        if "time" in df.columns:
            times = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
            times = np.where(np.isnan(times), np.arange(len(df)) * epoch_seconds, times)
        else:
            times = np.arange(len(df), dtype=float) * epoch_seconds
        grid = float(common_seconds) if common_seconds > 0 else 30.0
        out["epoch"] = np.floor((times + 1e-6) / grid).astype(int)
        grouped = (
            out.groupby("epoch")["stage"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
            .reset_index()
            .rename(columns={"stage": column})
        )
        return grouped

    def save_scoring_csv(self, path: str) -> bool:
        """Export current scoring to CSV."""
        df = self.sleep_scoring_data
        if df is None or df.empty:
            return False
        try:
            from CESA.scoring_io import export_scoring_csv
            export_scoring_csv(
                df,
                path,
                epoch_seconds=self.scoring_epoch_duration,
                night_start_s=self.night_start_s,
                night_end_s=self.night_end_s,
            )
            logger.info("[SCORING] Saved %d epochs to %s", len(df), path)
            return True
        except Exception as exc:
            logger.error("Scoring CSV export failed: %s", exc)
            return False

    def save_scoring_json(self, path: str) -> bool:
        """Export current scoring to JSON."""
        df = self.sleep_scoring_data
        if df is None or df.empty:
            return False
        try:
            from CESA.scoring_io import export_scoring_json
            export_scoring_json(
                df,
                path,
                epoch_seconds=self.scoring_epoch_duration,
                night_start_s=self.night_start_s,
                night_end_s=self.night_end_s,
            )
            logger.info("[SCORING] Saved %d epochs to %s", len(df), path)
            return True
        except Exception as exc:
            logger.error("Scoring JSON export failed: %s", exc)
            return False

    def save_scoring_file(self, path: str) -> bool:
        """Export scoring to CSV or JSON based on file extension."""
        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            return self.save_scoring_json(path)
        return self.save_scoring_csv(path)

    # ------------------------------------------------------------------
    # Night analysis window
    # ------------------------------------------------------------------

    def set_night_window(self, start_s: float, end_s: float) -> bool:
        """Set analysis window (seconds from recording start)."""
        start_s = float(start_s)
        end_s = float(end_s)
        if start_s >= end_s:
            logger.warning("[NIGHT] Invalid window: start (%.1f) >= end (%.1f)", start_s, end_s)
            return False
        if self.total_duration_s > 0 and end_s > self.total_duration_s + 1e-6:
            end_s = float(self.total_duration_s)
        self.night_start_s = start_s
        self.night_end_s = end_s
        self.night_start_min = start_s / 60.0
        self.night_end_min = end_s / 60.0
        logger.info("[NIGHT] Window set: %.1f–%.1f s (%.1f–%.1f min)", start_s, end_s, self.night_start_min, self.night_end_min)
        return True

    def clear_night_window(self) -> None:
        """Remove analysis window filter."""
        self.night_start_s = None
        self.night_end_s = None
        self.night_start_min = None
        self.night_end_min = None

    @property
    def has_night_window(self) -> bool:
        return self.night_start_s is not None and self.night_end_s is not None

    def filter_scoring_to_night_window(self, df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """Return scoring DataFrame restricted to the night analysis window."""
        if df is None:
            df = self.sleep_scoring_data
        if df is None or df.empty or not self.has_night_window:
            return df
        start_s = float(self.night_start_s)  # type: ignore[arg-type]
        end_s = float(self.night_end_s)  # type: ignore[arg-type]
        ep = float(self.scoring_epoch_duration)
        times = pd.to_numeric(df["time"], errors="coerce")
        mask = (times >= start_s - 1e-6) & (times + ep <= end_s + 1e-6)
        filtered = df.loc[mask].copy().reset_index(drop=True)
        return filtered if len(filtered) else df.iloc[0:0].copy()

    def get_active_scoring_df(self) -> Optional[pd.DataFrame]:
        """Return active scoring, optionally filtered by night window."""
        df = self.sleep_scoring_data
        if df is None or df.empty:
            return None
        return self.filter_scoring_to_night_window(df)

    def detect_events(
        self,
        event_type: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Detect sleep events. Returns list of event dicts.

        *event_type* is one of ``"arousals"``, ``"apneas"``, ``"desaturations"``.
        """
        if self.raw is None:
            return []
        if progress_callback:
            progress_callback(f"Detection: {event_type}...")
        try:
            from CESA.sleep_pipeline.events import (
                detect_arousals,
                detect_apneas_hypopneas,
                detect_desaturations,
            )
            from CESA.sleep_pipeline.contracts import ScoringResult  # noqa: F401 (not used directly)

            data = self.raw.get_data()
            sfreq = self.sfreq
            ch_names = list(self.raw.ch_names)

            if event_type == "arousals":
                eeg_idx = next(
                    (i for i, n in enumerate(ch_names)
                     if cesa_detect_signal_type(n) == "eeg"), 0)
                events = detect_arousals(data[eeg_idx], sfreq)
            elif event_type == "apneas":
                flow_idx = next(
                    (i for i, n in enumerate(ch_names)
                     if "flow" in n.lower() or "airflow" in n.lower()
                     or "nasal" in n.lower()), None)
                if flow_idx is None:
                    logger.warning("[EVENTS] No airflow channel found")
                    return []
                events = detect_apneas_hypopneas(data[flow_idx], sfreq)
            elif event_type == "desaturations":
                spo2_idx = next(
                    (i for i, n in enumerate(ch_names)
                     if "spo2" in n.lower() or "sao2" in n.lower()
                     or "ox" in n.lower()), None)
                if spo2_idx is None:
                    logger.warning("[EVENTS] No SpO2 channel found")
                    return []
                events = detect_desaturations(data[spo2_idx], sfreq)
            else:
                logger.warning("[EVENTS] Unknown event type: %s", event_type)
                return []

            result = []
            for e in events:
                if hasattr(e, "to_dict"):
                    result.append(e.to_dict())
                elif isinstance(e, dict):
                    d = dict(e)
                    if "onset" not in d and "onset_s" in d:
                        d["onset"] = d["onset_s"]
                    if "duration" not in d and "duration_s" in d:
                        d["duration"] = d["duration_s"]
                    if "type" not in d and "event_type" in d:
                        d["type"] = d["event_type"]
                    result.append(d)
                else:
                    logger.warning("[EVENTS] Skipping unscored event object: %r", type(e))
            logger.info("[EVENTS] %s: detected %d events", event_type, len(result))
            return result
        except Exception as exc:
            logger.error("Event detection failed (%s): %s", event_type, exc, exc_info=True)
            return []

    def _resolve_micro_channels(
        self,
        channels: Optional[Sequence[str]] = None,
        eog_channels: Optional[Sequence[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """Resolve the EEG / EOG channels used by micro-architecture analyses."""
        all_names = list(self.derivations.keys())
        types = {ch: cesa_detect_signal_type(ch) for ch in all_names}
        eeg_available = [c for c in all_names if types.get(c) == "eeg"]
        eog_available = [c for c in all_names if types.get(c) == "eog"]

        if channels:
            eeg_chs = [c for c in channels if c in self.derivations and types.get(c) == "eeg"]
            if not eeg_chs:
                # allow explicit names even if type detection failed
                eeg_chs = [c for c in channels if c in self.derivations]
        else:
            # Prefer YASA candidates present in the recording
            preferred = [c for c in self.yasa_eeg_candidates if c in eeg_available]
            eeg_chs = preferred or eeg_available[:4]

        if eog_channels:
            eog_chs = [c for c in eog_channels if c in self.derivations]
        else:
            preferred_eog = [c for c in self.yasa_eog_candidates if c in eog_available]
            eog_chs = preferred_eog or eog_available[:2]

        return eeg_chs, eog_chs

    def detect_so_spindle_coupling(
        self,
        channels: Optional[Sequence[str]] = None,
        *,
        use_hypnogram: bool = True,
        headband: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Classical SO-spindle coupling (filters + Hilbert + circular stats).

        Independent of YASA.  Returns a dict with keys ``events`` (UI dicts),
        ``coupling`` (per-channel preferred phase records), ``warnings``,
        ``used_hypnogram``, ``backend`` and ``eeg_channels``.

        With *headband*, only frontal derivations are analysed and ocular
        deflections are masked out, since a headband carries no EOG reference.
        """
        empty: Dict[str, Any] = {
            "events": [],
            "coupling": [],
            "warnings": [],
            "used_hypnogram": False,
            "backend": "none",
            "eeg_channels": [],
        }
        if not self.derivations:
            empty["warnings"].append("Aucun enregistrement charge.")
            return empty

        try:
            from CESA.sleep_pipeline.micro_events import (
                scored_events_to_dicts,
                scoring_df_to_hypno,
            )
            from CESA.sleep_pipeline.so_spindle_coupling import (
                detect_so_spindle_coupling as run_coupling,
            )
        except Exception as exc:
            empty["warnings"].append(f"Import so_spindle_coupling echoue: {exc}")
            return empty

        warnings: List[str] = []
        sfreq = float(self.sfreq)
        eeg_chs, _ = self._resolve_micro_channels(channels)
        if not eeg_chs:
            empty["warnings"].append("Aucun canal EEG disponible pour le couplage.")
            return empty

        if headband:
            from CESA.sleep_pipeline.headband_coupling import select_headband_channels

            try:
                eeg_chs, skipped = select_headband_channels(eeg_chs)
            except ValueError as exc:
                empty["warnings"].append(f"Montage bandeau: {exc}.")
                return empty
            if skipped:
                warnings.append(
                    "Montage bandeau: canaux non frontaux ignores — "
                    + ", ".join(skipped)
                )

        scoring_df = self.get_active_scoring_df() if use_hypnogram else None
        n_samples = int(self.derivations[eeg_chs[0]].shape[0])

        hypno = None
        if use_hypnogram and scoring_df is not None:
            hypno = scoring_df_to_hypno(
                scoring_df, n_samples, sfreq, self.scoring_epoch_duration,
            )
        elif use_hypnogram and scoring_df is None:
            warnings.append(
                "Aucun scoring disponible: couplage sans restriction de stade."
            )

        data = np.vstack([self.derivations[c] for c in eeg_chs])
        try:
            result = run_coupling(
                data, sfreq, eeg_chs,
                hypno=hypno,
                reject_blinks=headband,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            logger.error("SO-spindle coupling failed: %s", exc, exc_info=True)
            warnings.append(f"Couplage SO-fuseau: {exc}")
            return {**empty, "warnings": warnings, "eeg_channels": list(eeg_chs)}

        warnings.extend(result.warnings)
        coupling_records = (
            result.stats.to_dict(orient="records")
            if result.stats is not None and not result.stats.empty else []
        )
        return {
            "events": scored_events_to_dicts(result.events),
            "coupling": coupling_records,
            "warnings": warnings,
            "used_hypnogram": hypno is not None,
            "backend": result.backend,
            "eeg_channels": list(eeg_chs),
        }

    def detect_micro_events(
        self,
        kinds: Sequence[str],
        channels: Optional[Sequence[str]] = None,
        *,
        eog_channels: Optional[Sequence[str]] = None,
        use_hypnogram: bool = True,
        coupling: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Detect spindles / slow waves / REM bursts via YASA.

        Returns a dict with keys:
        ``events`` (list of UI dicts), ``summary`` (density records),
        ``pac`` (co-occurrence / PAC records), ``warnings`` (list of str),
        ``used_hypnogram`` (bool).
        """
        empty: Dict[str, Any] = {
            "events": [],
            "summary": [],
            "pac": [],
            "warnings": [],
            "used_hypnogram": False,
        }
        if not self.derivations:
            empty["warnings"].append("Aucun enregistrement charge.")
            return empty

        kinds_set = {str(k).strip().lower() for k in kinds}
        valid = {"spindles", "slow_waves", "rem_bursts"}
        kinds_set &= valid
        if not kinds_set:
            empty["warnings"].append("Aucun type d'evenement valide selectionne.")
            return empty

        def _progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)

        try:
            from CESA.sleep_pipeline.micro_events import (
                YasaUnavailableError,
                cooccurrence_summary,
                detect_rem_bursts,
                detect_slow_waves,
                detect_spindles,
                events_to_yasa_summary,
                scored_events_to_dicts,
                scoring_df_to_hypno,
                summarize_densities,
            )
        except Exception as exc:
            empty["warnings"].append(f"Import micro_events echoue: {exc}")
            return empty

        warnings: List[str] = []
        sfreq = float(self.sfreq)
        eeg_chs, eog_chs = self._resolve_micro_channels(channels, eog_channels)

        scoring_df = self.get_active_scoring_df() if use_hypnogram else None
        n_samples = 0
        if eeg_chs:
            n_samples = int(self.derivations[eeg_chs[0]].shape[0])
        elif eog_chs:
            n_samples = int(self.derivations[eog_chs[0]].shape[0])

        hypno = None
        used_hypno = False
        if use_hypnogram and scoring_df is not None and n_samples > 0:
            hypno = scoring_df_to_hypno(
                scoring_df, n_samples, sfreq, self.scoring_epoch_duration,
            )
            used_hypno = hypno is not None
        elif use_hypnogram and scoring_df is None:
            warnings.append(
                "Aucun scoring disponible: detection sans restriction de stade."
            )

        all_events = []
        spindle_events = []
        sw_events = []

        need_eeg = bool(kinds_set & {"spindles", "slow_waves"})
        if need_eeg and not eeg_chs:
            warnings.append("Aucun canal EEG disponible pour spindles / ondes lentes.")
        elif need_eeg:
            data = np.vstack([self.derivations[c] for c in eeg_chs])
            if "spindles" in kinds_set:
                _progress("Detection des spindles...")
                try:
                    spindle_events = detect_spindles(
                        data, sfreq, eeg_chs, hypno=hypno, include=(2,),
                    )
                    all_events.extend(spindle_events)
                except YasaUnavailableError as exc:
                    warnings.append(str(exc))
                    return {**empty, "warnings": warnings}
                except Exception as exc:
                    logger.error("Spindle detection failed: %s", exc, exc_info=True)
                    warnings.append(f"Spindles: {exc}")

            if "slow_waves" in kinds_set:
                _progress("Detection des ondes lentes (SO)...")
                try:
                    spindle_summary = (
                        events_to_yasa_summary(spindle_events)
                        if spindle_events else None
                    )
                    sw_events = detect_slow_waves(
                        data,
                        sfreq,
                        eeg_chs,
                        hypno=hypno,
                        include=(2, 3),
                        coupling=bool(coupling),
                        spindle_summary=spindle_summary,
                    )
                    all_events.extend(sw_events)
                except YasaUnavailableError as exc:
                    warnings.append(str(exc))
                    return {**empty, "warnings": warnings}
                except Exception as exc:
                    logger.error("Slow-wave detection failed: %s", exc, exc_info=True)
                    warnings.append(f"Ondes lentes: {exc}")

        if "rem_bursts" in kinds_set:
            if len(eog_chs) < 2:
                warnings.append(
                    "REM: deux canaux EOG (LOC/ROC) sont requis; "
                    f"disponibles={eog_chs or 'aucun'}."
                )
            else:
                _progress("Detection des mouvements oculaires rapides (REM)...")
                loc_name, roc_name = eog_chs[0], eog_chs[1]
                try:
                    rem_events = detect_rem_bursts(
                        self.derivations[loc_name],
                        self.derivations[roc_name],
                        sfreq,
                        hypno=hypno,
                        include=4,
                        loc_name=loc_name,
                        roc_name=roc_name,
                    )
                    all_events.extend(rem_events)
                except YasaUnavailableError as exc:
                    warnings.append(str(exc))
                    return {**empty, "warnings": warnings}
                except Exception as exc:
                    logger.error("REM detection failed: %s", exc, exc_info=True)
                    warnings.append(f"REM: {exc}")

        _progress("Calcul des densites...")
        summary_df = summarize_densities(
            all_events, scoring_df, self.scoring_epoch_duration,
        )
        pac_df = cooccurrence_summary(sw_events) if sw_events else None

        logger.info(
            "[MICRO] kinds=%s events=%d used_hypno=%s",
            sorted(kinds_set), len(all_events), used_hypno,
        )
        return {
            "events": scored_events_to_dicts(all_events),
            "summary": summary_df.to_dict(orient="records") if summary_df is not None else [],
            "pac": pac_df.to_dict(orient="records") if pac_df is not None and not pac_df.empty else [],
            "warnings": warnings,
            "used_hypnogram": used_hypno,
            "eeg_channels": list(eeg_chs),
            "eog_channels": list(eog_chs),
        }

    def import_scoring_file(self, path: str) -> bool:
        """Import scoring from CSV, Excel, or JSON."""
        try:
            from CESA.manual_scoring_service import ManualScoringService
            abs_start = None
            if self.raw is not None and self.raw.info.get("meas_date"):
                abs_start = pd.Timestamp(self.raw.info["meas_date"])
            result = ManualScoringService.import_file_path(
                path,
                absolute_start_datetime=abs_start,
                recording_duration_s=self.total_duration_s,
                default_epoch_seconds=self.scoring_epoch_duration,
            )
            df = result.df
            if df is not None and not df.empty:
                # Le scoring importe alimente uniquement la reference manuelle :
                # ecraser auto_scoring_data ici rendrait la comparaison egale a 100 %.
                self.manual_scoring_data = df
                self.manual_scoring_epoch_seconds = float(result.epoch_seconds)
                self.manual_scoring_source = os.path.basename(path)
                self.sleep_scoring_data = df.copy()
                self.active_scoring_source = "manual"
                self.scoring_epoch_duration = float(result.epoch_seconds)
                self.scoring_dirty = True
                if result.night_start_s is not None and result.night_end_s is not None:
                    self.set_night_window(float(result.night_start_s), float(result.night_end_s))
                logger.info("[SCORING] Imported scoring: %d epochs from %s", len(df), path)
                return True
        except Exception as exc:
            logger.error("Scoring import failed: %s", exc, exc_info=True)
        return False

    def import_scoring_excel(self, path: str) -> bool:
        """Import scoring from an Excel file and apply it as the active scoring."""
        return self.import_scoring_file(path)

    def import_scoring_edf(self, path: str) -> bool:
        """Import hypnogram from EDF+ annotations and apply it as the active scoring."""
        try:
            from CESA.scoring_io import import_edf_hypnogram
            abs_start = None
            if self.raw is not None and self.raw.info.get("meas_date"):
                abs_start = pd.Timestamp(self.raw.info["meas_date"])
            df = import_edf_hypnogram(
                path,
                recording_duration_s=self.total_duration_s,
                epoch_seconds=self.scoring_epoch_duration,
                absolute_start_datetime=abs_start,
            )
            if df is not None and not df.empty:
                self.manual_scoring_data = df
                self.manual_scoring_epoch_seconds = float(self.scoring_epoch_duration)
                self.manual_scoring_source = os.path.basename(path)
                self.sleep_scoring_data = df.copy()
                self.active_scoring_source = "manual"
                self.scoring_dirty = True
                logger.info("[SCORING] Imported EDF scoring: %d epochs from %s", len(df), path)
                return True
        except Exception as exc:
            logger.error("EDF scoring import failed: %s", exc, exc_info=True)
        return False

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def recording_file_filters_qt(self) -> str:
        """Return Qt-style file filter string."""
        exts = " ".join(f"*{e}" for e in SUPPORTED_RECORDING_EXTENSIONS)
        return f"Fichiers enregistrement ({exts});;Tous les fichiers (*.*)"

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        try:
            self._preprocess_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        if self._bridge_executor:
            try:
                self._bridge_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
