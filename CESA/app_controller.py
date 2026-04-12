"""Toolkit-independent application controller for CESA.

Holds all business state (loaded recording, filters, scoring, profiles,
navigation) so that the Qt main window can operate without any Tkinter
dependency.
"""

from __future__ import annotations

import logging
import os
import threading
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
from CESA.profile_schema import DisplayProcessingProfile, SignalSection
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
        self.autoscale_enabled: bool = False
        self.autoscale_window_duration: float = 30.0

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
        self.sleep_scoring_data: Optional[pd.DataFrame] = None
        self.manual_scoring_data: Optional[pd.DataFrame] = None
        self.sleep_scoring_method: str = "pftsleep"
        self.auto_scoring_epoch_length: float = 30.0
        self.scoring_epoch_duration: float = 30.0
        self.scoring_dirty: bool = False
        self.show_manual_scoring: bool = True

        self.night_start_min: Optional[float] = None
        self.night_end_min: Optional[float] = None

        # YASA-specific
        self.yasa_eeg_candidates: List[str] = [
            'C4-M1', 'F4-M1', 'C3-M2', 'F3-M2', 'O2-M1', 'O1-M2',
            'C4', 'F4', 'C3', 'F3', 'O2', 'O1',
        ]
        self.yasa_eog_candidates: List[str] = [
            'E2-M1', 'E1-M2', 'EOG', 'EOG gauche', 'EOG droite',
        ]
        self.yasa_emg_candidates: List[str] = ['EMG', 'EMG Chin', 'Chin', 'MENTON']
        self.yasa_target_sfreq: float = 100.0
        self.yasa_confidence_threshold: float = 0.80

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
        self.profile_store.save_profile(p)
        self.profile_store.set_last_profile_name(p.name)

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

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    SUPPORTED_SCORING_METHODS = (
        "pftsleep", "yasa", "usleep",
        "aasm_rules", "ml", "ml_hmm", "rules_hmm",
    )

    def run_auto_scoring(
        self,
        method: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Run automatic sleep scoring. Returns True on success."""
        if self.raw is None:
            return False
        selected = (method or self.sleep_scoring_method).lower().strip()
        if selected not in self.SUPPORTED_SCORING_METHODS:
            selected = "pftsleep"
        self.sleep_scoring_method = selected

        if progress_callback:
            progress_callback(f"Scoring automatique ({selected})...")

        try:
            from CESA.sleep_scorer import SleepScorer
            scorer = SleepScorer(
                method=selected,
                epoch_length=self.auto_scoring_epoch_length,
                eeg_candidates=tuple(self.yasa_eeg_candidates),
                eog_candidates=tuple(self.yasa_eog_candidates),
                emg_candidates=tuple(self.yasa_emg_candidates),
                target_sfreq=self.yasa_target_sfreq,
                yasa_confidence_threshold=self.yasa_confidence_threshold,
            )
            df = scorer.score(self.raw)
            if df is not None and not df.empty:
                self.sleep_scoring_data = df
                self.scoring_epoch_duration = float(self.auto_scoring_epoch_length)
                self.scoring_dirty = True
                logger.info("[SCORING] %s: %d epochs", selected, len(df))
                return True
        except Exception as exc:
            logger.error("Auto scoring failed: %s", exc, exc_info=True)
        return False

    def compare_scoring(self) -> Optional[pd.DataFrame]:
        """Compare auto vs manual scoring. Returns concordance DataFrame or None."""
        auto = self.sleep_scoring_data
        manual = self.manual_scoring_data
        if auto is None or manual is None or auto.empty or manual.empty:
            return None
        try:
            epoch_len = self.scoring_epoch_duration
            auto_c = auto.copy()
            manual_c = manual.copy()
            auto_c['epoch'] = np.floor((auto_c['time'] + 1e-6) / epoch_len).astype(int)
            manual_c['epoch'] = np.floor((manual_c['time'] + 1e-6) / epoch_len).astype(int)
            merged = pd.merge(
                auto_c[['epoch', 'stage']].rename(columns={'stage': 'auto'}),
                manual_c[['epoch', 'stage']].rename(columns={'stage': 'manual'}),
                on='epoch', how='inner',
            )
            merged['match'] = merged['auto'] == merged['manual']
            return merged
        except Exception as exc:
            logger.error("Scoring comparison failed: %s", exc)
            return None

    def save_scoring_csv(self, path: str) -> bool:
        """Export current scoring to CSV."""
        df = self.sleep_scoring_data
        if df is None or df.empty:
            return False
        try:
            df.to_csv(path, index=False)
            logger.info("[SCORING] Saved %d epochs to %s", len(df), path)
            return True
        except Exception as exc:
            logger.error("Scoring CSV export failed: %s", exc)
            return False

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
            from CESA.sleep_pipeline.contracts import ScoringResult, EpochScoring

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

            result = [e.to_dict() if hasattr(e, 'to_dict') else dict(e) for e in events]
            logger.info("[EVENTS] %s: detected %d events", event_type, len(result))
            return result
        except Exception as exc:
            logger.error("Event detection failed (%s): %s", event_type, exc, exc_info=True)
            return []

    def import_scoring_excel(self, path: str) -> bool:
        """Import scoring from an Excel file."""
        try:
            from CESA.scoring_io import import_excel_scoring
            df = import_excel_scoring(pd.read_excel(path))
            if df is not None and not df.empty:
                self.manual_scoring_data = df
                self.scoring_dirty = True
                return True
        except Exception as exc:
            logger.error("Excel scoring import failed: %s", exc)
        return False

    def import_scoring_edf(self, path: str) -> bool:
        """Import hypnogram from EDF+ annotations."""
        try:
            from CESA.scoring_io import import_edf_hypnogram
            df = import_edf_hypnogram(path)
            if df is not None and not df.empty:
                self.manual_scoring_data = df
                self.scoring_dirty = True
                return True
        except Exception as exc:
            logger.error("EDF scoring import failed: %s", exc)
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
