"""Header-only EDF/BDF/FIF reader and preview chunk loader.

Reads file headers via MNE (fast, ``preload=False``) and pre-loads a
short preview window (``preload=True``) for the signal preview in the
import wizard.  A lightweight direct EDF reader supplements per-channel
metadata (physical range, digital range, true unit).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PREVIEW_DURATION_S = 30.0  # seconds of data cached for preview
PREVIEW_SKIP_S = 30.0      # skip the first N seconds (often flat/calibration)


# ======================================================================
# Lightweight EDF header reader (metadata only, no MNE dependency)
# ======================================================================

class _EDFHeaderInfo:
    """Parse the binary EDF header to extract per-signal metadata.

    This is used to provide accurate physical/digital range and unit
    information, not for reading signal data.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.labels: List[str] = []
        self.n_signals: int = 0
        self.sfreq_per_signal: List[float] = []
        self.physical_min: List[float] = []
        self.physical_max: List[float] = []
        self.digital_min: List[int] = []
        self.digital_max: List[int] = []
        self.samples_per_record: List[int] = []
        self.physical_dimension: List[str] = []
        self.n_data_records: int = 0
        self.data_record_duration: float = 1.0
        self.header_bytes: int = 0
        self._ok = False
        self._parse()

    @property
    def ok(self) -> bool:
        return self._ok

    def _parse(self) -> None:
        try:
            with open(self.path, "rb") as f:
                main = f.read(256)
                if len(main) < 256:
                    return
                self.n_signals = int(main[252:256].decode("ascii", "replace").strip())
                self.n_data_records = int(main[236:244].decode("ascii", "replace").strip())
                self.data_record_duration = float(main[244:252].decode("ascii", "replace").strip())
                self.header_bytes = int(main[184:192].decode("ascii", "replace").strip())

                ns = self.n_signals
                sh = f.read(ns * 256)
                if len(sh) < ns * 256:
                    return
                off = 0
                self.labels = self._f(sh, off, 16, ns); off += 16 * ns
                off += 80 * ns  # transducer
                self.physical_dimension = self._f(sh, off, 8, ns); off += 8 * ns
                self.physical_min = [float(v) for v in self._f(sh, off, 8, ns)]; off += 8 * ns
                self.physical_max = [float(v) for v in self._f(sh, off, 8, ns)]; off += 8 * ns
                self.digital_min = [int(v) for v in self._f(sh, off, 8, ns)]; off += 8 * ns
                self.digital_max = [int(v) for v in self._f(sh, off, 8, ns)]; off += 8 * ns
                off += 80 * ns  # prefiltering
                self.samples_per_record = [int(v) for v in self._f(sh, off, 8, ns)]
                dur = self.data_record_duration if self.data_record_duration > 0 else 1.0
                self.sfreq_per_signal = [s / dur for s in self.samples_per_record]
            self._ok = True
            logger.info("[EDF-HDR] %d signals, labels=%s", ns, self.labels)
        except Exception as exc:
            logger.warning("[EDF-HDR] Parse failed: %s", exc)

    @staticmethod
    def _f(buf: bytes, offset: int, width: int, count: int) -> List[str]:
        return [
            buf[offset + i * width: offset + (i + 1) * width]
            .decode("ascii", "replace").strip()
            for i in range(count)
        ]

    def resolve_index(self, channel_name: str) -> int:
        """Find signal index with fuzzy matching. Returns -1 if not found."""
        if channel_name in self.labels:
            return self.labels.index(channel_name)
        cn = channel_name.lower().strip()
        for i, lbl in enumerate(self.labels):
            if lbl.lower().strip() == cn:
                return i
        for i, lbl in enumerate(self.labels):
            if cn in lbl.lower():
                return i
        for i, lbl in enumerate(self.labels):
            ll = lbl.lower().strip()
            if ll and ll in cn:
                return i
        return -1

    def info_for(self, channel_name: str) -> Dict[str, str]:
        """Return metadata dict for *channel_name*, or empty dict."""
        idx = self.resolve_index(channel_name)
        if idx < 0:
            return {}
        return {
            "edf_label": self.labels[idx],
            "unit": self.physical_dimension[idx] if idx < len(self.physical_dimension) else "",
            "sfreq": f"{self.sfreq_per_signal[idx]:.1f}" if idx < len(self.sfreq_per_signal) else "",
            "phys_min": f"{self.physical_min[idx]:.2f}" if idx < len(self.physical_min) else "",
            "phys_max": f"{self.physical_max[idx]:.2f}" if idx < len(self.physical_max) else "",
            "dig_min": str(self.digital_min[idx]) if idx < len(self.digital_min) else "",
            "dig_max": str(self.digital_max[idx]) if idx < len(self.digital_max) else "",
            "n_samples_per_record": str(self.samples_per_record[idx]) if idx < len(self.samples_per_record) else "",
            "n_samples_total": str(self.samples_per_record[idx] * self.n_data_records) if idx < len(self.samples_per_record) else "",
        }


# ======================================================================
# Main loader class
# ======================================================================

class EDFMetadataLoader:
    """Read recording metadata and cache a short preview chunk.

    The preview is loaded using the exact same method as the main viewer
    (``preload=True``) so it is guaranteed to produce real data.
    """

    def __init__(self) -> None:
        self._raw = None
        self._file_path: Optional[Path] = None
        self._edf_hdr: Optional[_EDFHeaderInfo] = None
        # Cached preview: {ch_name: 1-D numpy array (µV)}
        self._preview_cache: Dict[str, np.ndarray] = {}
        self._preview_sfreq: float = 0.0
        self._preview_duration: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_header(self, path: str | Path) -> "EDFSession | None":
        """Open *path*, read header, and cache a short preview chunk."""
        from .edf_session import ChannelInfo, DEFAULT_GAINS, EDFSession

        path = Path(path)
        if not path.exists():
            logger.error("Fichier introuvable : %s", path)
            return None

        # 1. Fast header-only open (for metadata)
        try:
            raw = self._open_raw(path)
        except Exception as exc:
            logger.error("Impossible d'ouvrir le fichier : %s", exc)
            return None

        self._raw = raw
        self._file_path = path

        # 2. Parse the binary EDF header for accurate per-channel info
        self._edf_hdr = _EDFHeaderInfo(path)

        sfreq = float(raw.info["sfreq"])
        n_times = raw.n_times
        duration_s = n_times / sfreq if sfreq > 0 else 0.0

        if self._edf_hdr.ok:
            logger.info(
                "[HEADER] MNE names=%s | EDF labels=%s",
                list(raw.ch_names[:6]), self._edf_hdr.labels[:6],
            )

        # 3. Build channel list
        channels: List[ChannelInfo] = []
        for idx, ch_name in enumerate(raw.ch_names):
            ch_info = raw.info["chs"][idx]
            sig_type = self._detect_type(ch_name)

            edf_info = self._edf_hdr.info_for(ch_name) if self._edf_hdr.ok else {}
            if edf_info.get("unit"):
                unit = edf_info["unit"]
            else:
                unit = self._unit_from_mne(ch_info)

            if edf_info.get("phys_min") and edf_info.get("phys_max"):
                phys_min = float(edf_info["phys_min"])
                phys_max = float(edf_info["phys_max"])
            else:
                phys_min, phys_max = self._physical_range(ch_info)

            channels.append(
                ChannelInfo(
                    name=ch_name,
                    signal_type=sig_type,
                    sfreq=sfreq,
                    physical_min=phys_min,
                    physical_max=phys_max,
                    unit=unit,
                    n_samples=n_times,
                    selected=sig_type != "other",
                    gain=DEFAULT_GAINS.get(sig_type, DEFAULT_GAINS["other"]),
                )
            )

        patient_info = self._extract_patient_info(raw)
        recording_date = self._extract_date(raw)

        session = EDFSession(
            file_path=path,
            channels=channels,
            sfreq=sfreq,
            duration_s=duration_s,
            n_samples=n_times,
            patient_info=patient_info,
            recording_date=recording_date,
        )
        logger.info(
            "[HEADER] %s : %d canaux, %.1f Hz, %s",
            path.name, len(channels), sfreq, session.duration_hms,
        )

        # 4. Cache preview data (preload=True, same as main viewer)
        self._cache_preview(path, sfreq)

        return session

    def load_preview_chunk(
        self,
        channels: List[str],
        start_s: float = 0.0,
        duration_s: float = 5.0,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, str]]]:
        """Return cached preview data for the requested channels.

        *start_s* is relative to the beginning of the cached window
        (not the beginning of the file).

        Returns ``{ch_name: (times, data_µV, info_dict)}``.
        """
        if not self._preview_cache or self._preview_sfreq <= 0:
            logger.warning("[PREVIEW] No cached data available")
            return {}

        sfreq = self._preview_sfreq
        offset = getattr(self, "_preview_offset", 0.0)
        result: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, str]]] = {}

        for ch_name in channels:
            if ch_name not in self._preview_cache:
                logger.warning("[PREVIEW] '%s' not in cache. Available: %s",
                               ch_name, list(self._preview_cache.keys())[:5])
                continue

            full_arr = self._preview_cache[ch_name]
            s0 = max(0, int(start_s * sfreq))
            s1 = min(len(full_arr), int((start_s + duration_s) * sfreq))
            if s1 <= s0:
                continue

            data = full_arr[s0:s1].copy()
            times = np.arange(len(data)) / sfreq + offset + start_s

            info = self._edf_hdr.info_for(ch_name) if self._edf_hdr and self._edf_hdr.ok else {}
            result[ch_name] = (times, data, info)

        return result

    def close(self) -> None:
        if self._raw is not None:
            try:
                self._raw.close()
            except Exception:
                pass
            self._raw = None
        self._preview_cache.clear()
        self._edf_hdr = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_file(path: str | Path) -> Tuple[bool, str]:
        from core.raw_loader import SUPPORTED_RECORDING_EXTENSIONS, normalize_recording_extension

        p = Path(path)
        if not p.exists():
            return False, "Le fichier n'existe pas."
        if not p.is_file():
            return False, "Le chemin ne pointe pas vers un fichier."
        try:
            ext = normalize_recording_extension(p)
        except Exception:
            ext = p.suffix.lower()
        if ext not in SUPPORTED_RECORDING_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_RECORDING_EXTENSIONS))
            return False, f"Format non supporte ({p.suffix}). Formats acceptes : {supported}"
        if p.stat().st_size < 256:
            return False, "Le fichier est trop petit pour etre un enregistrement valide."
        return True, "Fichier valide."

    # ------------------------------------------------------------------
    # Preview cache
    # ------------------------------------------------------------------

    def _cache_preview(self, path: Path, sfreq: float) -> None:
        """Load a short segment with ``preload=True`` and cache per-channel.

        Skips the first ``PREVIEW_SKIP_S`` seconds (often empty / calibration)
        and caches ``PREVIEW_DURATION_S`` seconds starting from that offset.
        """
        self._preview_cache.clear()
        self._preview_sfreq = sfreq
        self._preview_offset = 0.0

        if sfreq <= 0:
            return

        try:
            from core.raw_loader import open_raw_file
            logger.info("[PREVIEW-CACHE] Loading file with preload=True ...")
            raw_full = open_raw_file(str(path), preload=True, verbose=False)

            total_samples = raw_full.n_times
            total_dur = total_samples / sfreq

            # Start at the middle of the recording for representative data
            skip = max(0.0, (total_dur - PREVIEW_DURATION_S) / 2.0)

            start_sample = int(skip * sfreq)
            end_sample = min(total_samples, start_sample + int(PREVIEW_DURATION_S * sfreq))
            n_preview = end_sample - start_sample
            self._preview_duration = n_preview / sfreq
            self._preview_offset = skip

            data = raw_full.get_data(start=start_sample, stop=end_sample)
            ch_names = list(raw_full.ch_names)

            logger.info(
                "[PREVIEW-CACHE] %d ch x %d samples (%.1fs from t=%.0fs). "
                "shape=%s, dtype=%s",
                len(ch_names), n_preview, self._preview_duration, skip,
                data.shape, data.dtype,
            )

            for i, ch_name in enumerate(ch_names):
                arr = np.asarray(data[i], dtype=np.float64)

                edf_info = self._edf_hdr.info_for(ch_name) if self._edf_hdr and self._edf_hdr.ok else {}
                unit = edf_info.get("unit", "")
                arr = self._normalise_to_uv(arr, unit)

                np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                self._preview_cache[ch_name] = arr

                if i < 3:
                    logger.info(
                        "[PREVIEW-CACHE] %s: min=%.4g max=%.4g std=%.4g (unit='%s')",
                        ch_name, float(arr.min()), float(arr.max()),
                        float(arr.std()), unit,
                    )

            del raw_full
            logger.info("[PREVIEW-CACHE] Done. %d channels cached.", len(self._preview_cache))

        except Exception as exc:
            logger.error("[PREVIEW-CACHE] Failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _open_raw(path: Path):
        from core.raw_loader import open_raw_file
        return open_raw_file(str(path), preload=False, verbose=False)

    @staticmethod
    def _normalise_to_uv(arr: np.ndarray, unit_str: str) -> np.ndarray:
        """Convert to µV based on declared unit, with heuristic fallback."""
        if arr.size == 0:
            return arr
        u = unit_str.strip().lower()
        if u in ("uv", "µv", "microv", "microvolt", "microvolts"):
            return arr
        if u in ("mv", "millivolt", "millivolts"):
            return arr * 1e3
        if u in ("v", "volt", "volts"):
            return arr * 1e6
        # Unknown unit — heuristic based on magnitude
        abs_max = float(np.nanmax(np.abs(arr)))
        if abs_max == 0.0:
            return arr
        if abs_max < 0.1:
            return arr * 1e6
        if abs_max < 100.0:
            return arr * 1e3
        return arr

    @staticmethod
    def _detect_type(channel_name: str) -> str:
        try:
            from CESA.filters import detect_signal_type
            raw_type = detect_signal_type(channel_name)
            if raw_type in ("eeg", "eog", "emg", "ecg"):
                return raw_type
            if raw_type == "sas_eeg":
                return "eeg"
            if raw_type == "sas_emg":
                return "emg"
            return "other"
        except Exception:
            return "other"

    @staticmethod
    def _unit_from_mne(ch_info: dict) -> str:
        unit_code = ch_info.get("unit", 0)
        if unit_code == 107:
            cal = ch_info.get("cal", 1.0)
            if abs(cal) < 1e-3:
                return "µV"
            return "V"
        return "µV"

    @staticmethod
    def _physical_range(ch_info: dict) -> Tuple[float, float]:
        r = ch_info.get("range", 1.0)
        cal = ch_info.get("cal", 1.0)
        scale = abs(r * cal) if r and cal else 1.0
        return (-scale * 1e6, scale * 1e6)

    @staticmethod
    def _extract_patient_info(raw) -> Dict[str, str]:
        info: Dict[str, str] = {}
        subj = raw.info.get("subject_info")
        if isinstance(subj, dict):
            for key in ("his_id", "first_name", "last_name", "sex", "birthday"):
                val = subj.get(key)
                if val is not None:
                    info[key] = str(val)
        desc = raw.info.get("description")
        if desc:
            info["description"] = str(desc)
        return info

    @staticmethod
    def _extract_date(raw):
        meas_date = raw.info.get("meas_date")
        if meas_date is not None:
            from datetime import datetime, timezone
            if hasattr(meas_date, "timestamp"):
                return meas_date
            try:
                return datetime.fromtimestamp(float(meas_date), tz=timezone.utc)
            except Exception:
                pass
        return None
