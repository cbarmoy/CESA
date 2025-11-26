"""Centralised telemetry helpers for profiling viewer performance."""

from __future__ import annotations

import atexit
import csv
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

try:  # pragma: no cover - optional dependency for richer metrics
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil is optional
    psutil = None  # type: ignore


_DEFAULT_HEADER: tuple[str, ...] = (
    "timestamp",
    "dataset_id",
    "channel",
    "start_s",
    "duration_s",
    "viewport_px",
    "spp_screen",
    "level_k",
    "chunks_read",
    "bytes_read",
    "io_ms",
    "decim_ms",
    "filter_ms",
    "baseline_ms",  # Correction de baseline
    "render_ms",
    "total_ms",
    "cache_hit",
    "fps",
    "cpu_pct",
    "rss_mb",
    "notes",
    # Nouveaux champs pour le diagnostic de visualisation
    "load_edf_ms",          # Temps de chargement EDF
    "prepare_data_ms",      # Temps de préparation initiale des données
    "extract_channels_ms", # Temps d'extraction des canaux depuis MNE Raw
    "convert_uv_ms",        # Temps de conversion vers microvolts
    "prepare_hypno_ms",     # Temps de préparation de l'hypnogramme
    "create_plotter_ms",    # Temps de création de l'objet PSGPlotter
)


@dataclass
class TelemetryConfig:
    profile_io: bool = False
    profile_render: bool = False
    track_fps: bool = False
    csv_path: Path = field(default_factory=lambda: Path("logs/telemetry.csv"))
    reset_on_start: bool = False  # Si True, supprime le fichier CSV au démarrage


class TelemetryRecorder:
    """Thread-safe registry that aggregates and persists telemetry samples."""

    def __init__(self) -> None:
        self._config = TelemetryConfig()
        self._enabled = False
        self._dataset_id: str = "unknown"

        self._lock = threading.Lock()
        self._buffer: list[Dict[str, object]] = []
        self._flush_every = 64
        self._header_written = False
        self._expected_modes: set[str] = set()
        self._seen_modes: set[str] = set()

        self._process: Optional["psutil.Process"] = None  # type: ignore[name-defined]
        if psutil is not None:  # pragma: no cover - best effort initialisation
            try:
                self._process = psutil.Process(os.getpid())
            except Exception:
                self._process = None

        atexit.register(lambda: self.flush(final=True))

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def configure(
        self,
        *,
        profile_io: bool = False,
        profile_render: bool = False,
        track_fps: bool = False,
        csv_path: str | os.PathLike[str] | None = None,
        reset_on_start: bool = False,
    ) -> None:
        """Enable/disable telemetry features and configure persistence."""

        self._config.profile_io = bool(profile_io)
        self._config.profile_render = bool(profile_render)
        self._config.track_fps = bool(track_fps)
        self._config.reset_on_start = bool(reset_on_start)
        if csv_path is not None:
            self._config.csv_path = Path(csv_path)

        # Réinitialiser le fichier CSV si demandé
        if self._config.reset_on_start and self._config.csv_path.exists():
            try:
                self._config.csv_path.unlink()
                self._header_written = False
            except Exception:
                pass

        self._enabled = any(
            (
                self._config.profile_io,
                self._config.profile_render,
                self._config.track_fps,
            )
        )
        self._seen_modes.clear()

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    def set_dataset_id(self, dataset_id: str | Path | None) -> None:
        if dataset_id is None:
            return
        self._dataset_id = str(dataset_id)

    # ------------------------------------------------------------------
    # Sample helpers
    # ------------------------------------------------------------------
    def new_sample(self, defaults: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        if not self._enabled:
            return {} if defaults is None else dict(defaults)

        row: Dict[str, object] = {name: "" for name in _DEFAULT_HEADER}
        if defaults:
            row.update(defaults)
        row.setdefault("dataset_id", self._dataset_id)
        row["_perf_start"] = time.perf_counter()
        return row

    @contextmanager
    def measure(self, sample: Dict[str, object] | None, field: str):
        if not self._enabled or not sample:
            yield
            return

        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            previous = float(sample.get(field) or 0.0)
            sample[field] = previous + elapsed_ms

    def commit(self, sample: Dict[str, object] | None) -> None:
        if not self._enabled or not sample:
            return

        perf_start = sample.pop("_perf_start", None)
        if perf_start is not None:
            sample.setdefault("total_ms", (time.perf_counter() - perf_start) * 1000.0)

        sample.setdefault("timestamp", time.time())
        sample.setdefault("cache_hit", False)

        if sample.get("cpu_pct") in ("", None) or sample.get("rss_mb") in ("", None):
            cpu_pct, rss_mb = self._collect_process_metrics()
            sample.setdefault("cpu_pct", cpu_pct)
            sample.setdefault("rss_mb", rss_mb)

        with self._lock:
            self._buffer.append(sample)
            if len(self._buffer) >= self._flush_every:
                self._flush_locked(final=False)

    def extend(self, samples: Iterable[Dict[str, object]]) -> None:
        for sample in samples:
            self.commit(sample)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def flush(self, *, final: bool = False) -> None:
        with self._lock:
            if self._enabled:
                self._flush_locked(final=final)
            else:
                self._validate_expected_modes(final=final)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _collect_process_metrics(self) -> tuple[float, float]:
        if self._process is None:
            return (-1.0, -1.0)
        try:  # pragma: no cover - depends on psutil availability
            cpu_pct = self._process.cpu_percent(interval=0.0)
            rss_mb = self._process.memory_info().rss / (1024 * 1024)
            return (float(cpu_pct), float(rss_mb))
        except Exception:
            return (-1.0, -1.0)

    def _get_all_fieldnames(self) -> list[str]:
        """Récupère tous les noms de champs possibles en combinant le header par défaut
        et tous les champs présents dans le buffer, pour une compatibilité ascendante."""
        all_fields = set(_DEFAULT_HEADER)
        
        # Ajouter tous les champs présents dans les samples du buffer
        for sample in self._buffer:
            all_fields.update(k for k in sample.keys() if not k.startswith('_'))
        
        # Lire les champs existants du fichier CSV si disponible
        csv_path = self._config.csv_path
        if csv_path.exists():
            try:
                with csv_path.open('r', encoding='utf-8') as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames:
                        all_fields.update(reader.fieldnames)
            except Exception:
                pass
        
        # Trier: d'abord les champs par défaut dans l'ordre, puis les autres
        ordered = [f for f in _DEFAULT_HEADER if f in all_fields]
        extra = sorted(all_fields - set(_DEFAULT_HEADER))
        return ordered + extra

    def _flush_locked(self, *, final: bool = False) -> None:
        if not self._buffer:
            return

        csv_path = self._config.csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Obtenir tous les noms de champs (incluant les nouveaux champs optionnels)
        all_fieldnames = self._get_all_fieldnames()
        
        write_header = not csv_path.exists() or not self._header_written
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_fieldnames, extrasaction='ignore')
            if write_header:
                writer.writeheader()
                self._header_written = True
            for sample in self._buffer:
                row = {}
                for name in all_fieldnames:
                    value = sample.get(name, "")
                    # Convertir les valeurs selon le type de champ attendu
                    if name in ["timestamp", "dataset_id", "channel", "notes", "action"]:
                        # Ces champs doivent être des chaînes - conversion explicite
                        if value is None or value == "":
                            row[name] = ""
                        else:
                            # S'assurer que c'est bien une chaîne, même si c'était un nombre
                            row[name] = str(value)
                    elif name == "cache_hit":
                        # Booléen -> 0 ou 1
                        if isinstance(value, bool):
                            row[name] = 1 if value else 0
                        elif value in ("", None):
                            row[name] = 0
                        else:
                            row[name] = int(value) if value else 0
                    else:
                        # Valeurs numériques
                        if value in ("", None):
                            row[name] = ""
                        else:
                            try:
                                # Essayer de convertir en float
                                row[name] = float(value)
                            except (ValueError, TypeError):
                                # Si ça ne marche pas, convertir en chaîne
                                row[name] = str(value)
                writer.writerow(row)

        self._buffer.clear()
        self._validate_expected_modes(final=final)

    # ------------------------------------------------------------------
    # Mode checkpoints
    # ------------------------------------------------------------------
    def expect_modes(self, modes: Optional[Iterable[str]]) -> None:
        with self._lock:
            if modes:
                normalised = {
                    str(mode).lower().strip()
                    for mode in modes
                    if str(mode or "").strip()
                }
                self._expected_modes = normalised
            else:
                self._expected_modes = set()
            self._seen_modes.clear()

    def mark_mode(self, mode: str, *, source: str | None = None) -> None:
        mode_norm = str(mode or "").lower().strip()
        if not mode_norm:
            return

        with self._lock:
            self._seen_modes.add(mode_norm)
            expected = set(self._expected_modes)

        if expected and mode_norm not in expected:
            src = f" ({source})" if source else ""
            raise RuntimeError(
                f"Mode checkpoint failed: observed '{mode_norm}'{src}\n"
                f"Expected modes: {sorted(expected)}.\n"
                "Ensure the expected calculation implementation (lazy/hybrid) is active."
            )

    def _validate_expected_modes(self, *, final: bool) -> None:
        if not final:
            return
        expected = set(self._expected_modes)
        if not expected:
            return
        missing = expected.difference(self._seen_modes)
        if missing:
            raise RuntimeError(
                "Mode checkpoint failed: expected modes were not observed "
                f"{sorted(missing)}.\nEnsure the expected calculation path reports its mode "
                "via telemetry.mark_mode()."
            )


# Global singleton used across the codebase
telemetry = TelemetryRecorder()


def is_enabled() -> bool:
    return telemetry is not None and telemetry._enabled


