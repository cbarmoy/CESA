"""
CESA v0.0beta2.0 - Scoring I/O Utilities
================================

Module d'import et de synchronisation du scoring pour CESA v0.0beta2.0.
Développé pour l'Unité Neuropsychologie du Stress (IRBA).

Fonctions pour normaliser le scoring manuel (Excel/CSV) et les annotations
hypnogramme EDF+. Format de sortie standardisé : DataFrame avec colonnes
['time','stage'] où 'time' est en secondes depuis le début de l'enregistrement.

Fonctionnalités principales:
- Import Excel/CSV avec détection automatique des colonnes
- Import hypnogramme EDF+ avec parsing des annotations
- Synchronisation temporelle robuste
- Validation des données et gestion des erreurs
- Support des formats de date/heure variés
- Cohérence des durées d'époques

Utilisé par l'interface CESA pour:
- Import du scoring manuel des techniciens
- Import automatique des hypnogrammes EDF+
- Synchronisation avec les données EEG
- Validation et correction des données

Formats supportés:
- Excel (.xls, .xlsx) avec colonnes configurables
- CSV avec séparateurs automatiques
- Annotations EDF+ standard (hypnogramme)
- Formats de date variés (ISO, français, etc.)

Auteur: Côme Barmoy (Unité Neuropsychologie du Stress - IRBA)
Version: 0.0beta2.0
Date: 2025-09-26
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union
import pandas as pd
import numpy as np
import mne

logger = logging.getLogger(__name__)

_META_LINE_RE = re.compile(r"^\s*#\s*(night_start_s|night_end_s|epoch_seconds|night_start_min|night_end_min)\s*=\s*([^\s#]+)\s*$")


@dataclass
class ScoringExportMeta:
    """Optional metadata stored alongside hypnogram exports."""

    epoch_seconds: Optional[float] = None
    night_start_s: Optional[float] = None
    night_end_s: Optional[float] = None


def parse_scoring_metadata_lines(lines: Iterable[str]) -> ScoringExportMeta:
    """Parse ``# key=value`` comment lines from a scoring export file."""
    meta = ScoringExportMeta()
    for raw in lines:
        line = raw.strip()
        if not line.startswith("#"):
            break
        match = _META_LINE_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        try:
            num = float(value)
        except ValueError:
            continue
        if key == "epoch_seconds":
            meta.epoch_seconds = num
        elif key == "night_start_s":
            meta.night_start_s = num
        elif key == "night_end_s":
            meta.night_end_s = num
        elif key == "night_start_min":
            meta.night_start_s = num * 60.0
        elif key == "night_end_min":
            meta.night_end_s = num * 60.0
    return meta


def read_scoring_csv_metadata(path: Union[str, Path]) -> ScoringExportMeta:
    text = Path(path).read_text(encoding="utf-8")
    return parse_scoring_metadata_lines(text.splitlines())


def _meta_from_json_document(data: dict) -> ScoringExportMeta:
    epoch = data.get("epoch_seconds")
    start = data.get("night_start_s")
    end = data.get("night_end_s")
    if start is None and data.get("night_start_min") is not None:
        start = float(data["night_start_min"]) * 60.0
    if end is None and data.get("night_end_min") is not None:
        end = float(data["night_end_min"]) * 60.0
    return ScoringExportMeta(
        epoch_seconds=float(epoch) if epoch is not None else None,
        night_start_s=float(start) if start is not None else None,
        night_end_s=float(end) if end is not None else None,
    )


_STAGE_MAP = {
    "W": "W", "WK": "W", "WAKE": "W", "EVEIL": "W", "AWAKE": "W", "0": "W",
    "WAKEFULNESS": "W", "WACH": "W", "VEILLE": "W",
    "N1": "N1", "NREM1": "N1", "S1": "N1", "STAGE1": "N1", "1": "N1",
    "STADE1": "N1", "LEGER": "N1",
    "N2": "N2", "NREM2": "N2", "S2": "N2", "STAGE2": "N2", "2": "N2",
    "STADE2": "N2",
    "N3": "N3", "NREM3": "N3", "S3": "N3", "STAGE3": "N3", "SWS": "N3",
    "S4": "N3", "STAGE4": "N3", "3": "N3", "4": "N3", "N4": "N3",
    "STADE3": "N3", "STADE4": "N3", "PROFOND": "N3",
    "R": "R", "REM": "R", "PARADOXAL": "R", "5": "R", "SP": "R",
    "U": "U", "UNKNOWN": "U", "?": "U", "ART": "U", "ARTEFACT": "U",
    "MVT": "U", "MT": "U", "MOVEMENT": "U", "INCONNU": "U",
}


def _strip_accents(s: str) -> str:
    """Remove common French accents for fuzzy matching."""
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def _normalize_stage_label(raw: object) -> str:
    """Map any common stage representation to canonical W/N1/N2/N3/R/U."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "U"
    text = str(raw).strip()
    if not text or text in ("[]", "nan", "None", "?"):
        return "U"
    # Numeric codes from Excel (0.0, 1, 2, …)
    try:
        code = int(float(text))
        key = str(code)
        if key in _STAGE_MAP:
            return _STAGE_MAP[key]
    except (ValueError, TypeError):
        pass
    s = text.upper().replace(" ", "")
    hit = _STAGE_MAP.get(s)
    if hit:
        return hit
    s_ascii = _strip_accents(s)
    hit = _STAGE_MAP.get(s_ascii)
    if hit:
        return hit
    for key, val in _STAGE_MAP.items():
        if key in s_ascii or s_ascii in key:
            return val
    return "U"


#: Canonical stage labels produced by :func:`normalize_stage_label`.
CANONICAL_STAGES = ("W", "N1", "N2", "N3", "R", "U")


def normalize_stage_label(raw: object) -> str:
    """Map any stage representation to canonical W/N1/N2/N3/R/U.

    Public wrapper over the importer's internal mapping so that consumers
    comparing two hypnograms (e.g. ``AppController.compare_scoring``) use the
    exact same conventions as the import path: ``REM`` -> ``R``,
    ``WAKE``/``Eveil`` -> ``W``, numeric codes, artefact/movement -> ``U``.
    """
    return _normalize_stage_label(raw)


def normalize_stage_series(values: Any) -> pd.Series:
    """Vectorised :func:`normalize_stage_label` over a Series or sequence."""
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    return series.map(_normalize_stage_label).astype(str)


def _is_cesa_native_format(df: pd.DataFrame) -> bool:
    """Return True when columns are explicitly ``time`` + ``stage`` with numeric seconds."""
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    if "time" not in cols_lower or "stage" not in cols_lower:
        return False
    times = pd.to_numeric(df[cols_lower["time"]], errors="coerce")
    return float(times.notna().mean()) >= 0.7


def _is_calendar_datetime_series(series: pd.Series) -> bool:
    """Distinguish real calendar datetimes from epoch-1970 false positives."""
    as_dt = pd.to_datetime(series, errors="coerce")
    if len(as_dt) == 0:
        return False
    valid = as_dt.notna()
    if float(valid.mean()) < 0.7:
        return False
    years = as_dt[valid].dt.year
    return float((years >= 2000).mean()) >= 0.7


def import_cesa_native_scoring(
    df: pd.DataFrame,
    epoch_seconds: float = 30.0,
    recording_duration_s: float = 0.0,
) -> pd.DataFrame:
    """Import CESA-exported scoring with columns ``time`` (seconds) and ``stage``."""
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    cols_lower = {c.lower(): c for c in work.columns}
    time_col = cols_lower.get("time", work.columns[0])
    stage_col = cols_lower.get("stage", work.columns[1] if len(work.columns) > 1 else work.columns[0])

    out = pd.DataFrame({
        "time": pd.to_numeric(work[time_col], errors="coerce"),
        "stage": work[stage_col].apply(_normalize_stage_label),
    }).dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    ep = float(epoch_seconds)
    out["time"] = (np.round(out["time"] / ep) * ep)
    out = out.drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)

    if recording_duration_s > 0 and len(out):
        last_t = float(out["time"].iloc[-1])
        t = last_t + ep
        max_dur = min(recording_duration_s, 24 * 3600)
        trailing = []
        while t < max_dur - 1e-6:
            trailing.append({"time": t, "stage": "U"})
            t += ep
        if trailing:
            out = pd.concat([out, pd.DataFrame(trailing)], ignore_index=True)

    return out


def import_scoring_table(
    df: pd.DataFrame,
    absolute_start_datetime: Optional[pd.Timestamp] = None,
    epoch_seconds: float = 30.0,
    recording_duration_s: float = 0.0,
) -> pd.DataFrame:
    """Normalize any supported scoring table to ``['time','stage']`` in seconds.

    Supports:
    - CESA native: ``time`` + ``stage`` (seconds from recording start)
    - Clinical Excel: ``stage`` + calendar datetime
    - Keyword-based column detection (via first-two-column fallback)
    """
    if _is_cesa_native_format(df):
        logger.info("[SCORING_IO] Detected CESA native format (time, stage)")
        return import_cesa_native_scoring(
            df,
            epoch_seconds=epoch_seconds,
            recording_duration_s=recording_duration_s,
        )

    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    cols_lower = {c.lower(): c for c in work.columns}

    stage_col: Optional[str] = None
    time_col: Optional[str] = None
    for col in work.columns:
        low = col.lower()
        if stage_col is None and any(k in low for k in ("stage", "stade", "sleep", "sommeil", "state", "score")):
            stage_col = col
        if time_col is None and any(k in low for k in ("time", "temps", "datetime", "date", "heure", "epoch", "epoque", "start", "debut")):
            time_col = col

    if stage_col is None:
        stage_col = work.columns[0]
    if time_col is None:
        time_col = work.columns[1] if len(work.columns) > 1 else work.columns[0]
    if stage_col == time_col and len(work.columns) > 1:
        time_col = work.columns[1]

    time_series = work[time_col]
    if _is_calendar_datetime_series(time_series):
        tmp = pd.DataFrame({
            "stage": work[stage_col],
            "datetime": pd.to_datetime(time_series, errors="coerce"),
        }).dropna(subset=["datetime", "stage"])
        return import_excel_scoring(
            tmp,
            absolute_start_datetime=absolute_start_datetime,
            epoch_seconds=epoch_seconds,
            recording_duration_s=recording_duration_s,
        )

    times = pd.to_numeric(time_series, errors="coerce")
    out = pd.DataFrame({
        "time": times,
        "stage": work[stage_col].apply(_normalize_stage_label),
    }).dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    ep = float(epoch_seconds)
    out["time"] = (np.round(out["time"] / ep) * ep)
    out = out.drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
    return out


def import_scoring_json(
    path: Union[str, Path],
    epoch_seconds: float = 30.0,
    recording_duration_s: float = 0.0,
) -> pd.DataFrame:
    """Load hypnogram from CESA JSON schema."""
    df, _meta = import_scoring_json_with_meta(
        path,
        epoch_seconds=epoch_seconds,
        recording_duration_s=recording_duration_s,
    )
    return df


def import_scoring_json_with_meta(
    path: Union[str, Path],
    epoch_seconds: float = 30.0,
    recording_duration_s: float = 0.0,
) -> Tuple[pd.DataFrame, ScoringExportMeta]:
    """Load hypnogram JSON and return epochs plus export metadata."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = ScoringExportMeta()
    if isinstance(data, dict) and "epochs" in data:
        meta = _meta_from_json_document(data)
        ep = float(meta.epoch_seconds if meta.epoch_seconds is not None else data.get("epoch_seconds", epoch_seconds))
        rows = data["epochs"]
    elif isinstance(data, list):
        ep = float(epoch_seconds)
        rows = data
    else:
        raise ValueError("Format JSON hypnogramme non reconnu.")

    df = pd.DataFrame(rows)
    if "time" not in df.columns or "stage" not in df.columns:
        raise ValueError("Le JSON doit contenir les champs 'time' et 'stage'.")
    out = import_cesa_native_scoring(df, epoch_seconds=ep, recording_duration_s=recording_duration_s)
    if meta.epoch_seconds is None:
        meta.epoch_seconds = ep
    return out, meta


def export_scoring_json(
    df: pd.DataFrame,
    path: Union[str, Path],
    epoch_seconds: float = 30.0,
    *,
    night_start_s: Optional[float] = None,
    night_end_s: Optional[float] = None,
) -> None:
    """Export hypnogram to CESA JSON schema."""
    out = df[["time", "stage"]].copy()
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["stage"] = out["stage"].apply(_normalize_stage_label)
    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    payload = {
        "epoch_seconds": float(epoch_seconds),
        "epochs": [
            {"time": float(row["time"]), "stage": str(row["stage"])}
            for _, row in out.iterrows()
        ],
    }
    if night_start_s is not None:
        payload["night_start_s"] = float(night_start_s)
    if night_end_s is not None:
        payload["night_end_s"] = float(night_end_s)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_scoring_csv(
    df: pd.DataFrame,
    path: Union[str, Path],
    *,
    epoch_seconds: float = 30.0,
    night_start_s: Optional[float] = None,
    night_end_s: Optional[float] = None,
) -> None:
    """Export hypnogram to CESA CSV format (time in seconds, stage)."""
    out = df[["time", "stage"]].copy()
    out["stage"] = out["stage"].apply(_normalize_stage_label)
    header_lines = [
        "# CESA hypnogram export",
        f"# epoch_seconds={float(epoch_seconds)}",
    ]
    if night_start_s is not None:
        header_lines.append(f"# night_start_s={float(night_start_s)}")
    if night_end_s is not None:
        header_lines.append(f"# night_end_s={float(night_end_s)}")
    csv_path = Path(path)
    csv_path.write_text("\n".join(header_lines) + "\n", encoding="utf-8")
    out.to_csv(csv_path, mode="a", index=False, encoding="utf-8")


def import_excel_scoring(
    df: pd.DataFrame,
    absolute_start_datetime: Optional[pd.Timestamp],
    epoch_seconds: float = 30.0,
    recording_duration_s: float = 0.0,
) -> pd.DataFrame:
    """Normalize Excel scoring DataFrame to columns ['time','stage'] in seconds.

    Times are snapped to the nearest epoch boundary.  Leading and trailing
    gaps are filled with 'U' when *recording_duration_s* is provided.
    """
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    stage_col, dt_col = work.columns[:2]
    work = work[[stage_col, dt_col]].dropna()
    work = work.rename(columns={stage_col: 'stage', dt_col: 'datetime'})
    work['datetime'] = pd.to_datetime(work['datetime'], errors='coerce')
    work = work.dropna(subset=['datetime', 'stage'])
    work = work[~work['stage'].astype(str).str.strip().isin(['[]', '', 'nan', 'None'])]

    if absolute_start_datetime is not None:
        base_ts = pd.Timestamp(absolute_start_datetime)
        series = work['datetime']
        if base_ts.tz is not None:
            if series.dt.tz is None:
                series = series.dt.tz_localize(base_ts.tz)
            else:
                series = series.dt.tz_convert(base_ts.tz)
        else:
            series = series.dt.tz_localize(None)

        times = (series - base_ts).dt.total_seconds()
        day_sec = 24.0 * 3600.0
        times = times.where(times >= 0.0, times + day_sec)
        work['time'] = times
    else:
        work = work.sort_values('datetime').reset_index(drop=True)
        work['time'] = np.arange(len(work), dtype=float) * float(epoch_seconds)

    # Snap each time to the nearest epoch boundary
    ep = float(epoch_seconds)
    work['time'] = (np.round(work['time'] / ep) * ep)

    # Normalize stage labels to canonical form (W/N1/N2/N3/R/U)
    raw_unique = work['stage'].unique().tolist()
    work['stage'] = work['stage'].apply(_normalize_stage_label)
    norm_unique = work['stage'].unique().tolist()
    logger.info("[SCORING_IO] Raw stage labels: %s -> Normalized: %s", raw_unique, norm_unique)

    out = work[['time', 'stage']].sort_values('time').reset_index(drop=True)
    out = out.drop_duplicates(subset=['time'], keep='first').reset_index(drop=True)

    # Fill leading epochs with 'U'
    if len(out) and float(out['time'].iloc[0]) > ep * 0.5:
        leading = []
        t = 0.0
        first_t = float(out['time'].iloc[0])
        while t < first_t - 1e-6:
            leading.append({'time': t, 'stage': 'U'})
            t += ep
        if leading:
            out = pd.concat([pd.DataFrame(leading), out], ignore_index=True)

    # Fill trailing epochs with 'U' up to recording duration
    if recording_duration_s > 0 and len(out):
        last_t = float(out['time'].iloc[-1])
        t = last_t + ep
        max_dur = min(recording_duration_s, 24 * 3600)
        trailing = []
        while t < max_dur - 1e-6:
            trailing.append({'time': t, 'stage': 'U'})
            t += ep
        if trailing:
            out = pd.concat([out, pd.DataFrame(trailing)], ignore_index=True)

    return out


def import_edf_hypnogram(
    edf_ann_path: str,
    recording_duration_s: float,
    epoch_seconds: float = 30.0,
    absolute_start_datetime: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Load EDF+ hypnogram annotations and return DataFrame ['time','stage'].

    Aligns annotations to the EDF+ main recording start when possible using
    orig_time. Fills leading and trailing 'U' epochs to cover full recording.
    """
    ann = mne.read_annotations(edf_ann_path)
    if ann is None or len(ann) == 0:
        return pd.DataFrame(columns=['time', 'stage'])

    rk_to_std = {'W': 'W', 'R': 'R', '1': 'N1', '2': 'N2', '3': 'N3', '4': 'N3', 'M': 'W', '?': 'U'}
    descs = [str(d) for d in ann.description]
    idxs, codes = [], []
    for i, d in enumerate(descs):
        if 'Sleep stage' in d:
            code = d.strip().split()[-1]
            idxs.append(i)
            codes.append(rk_to_std.get(code, 'U'))
    if not codes:
        return pd.DataFrame(columns=['time', 'stage'])

    onsets = np.asarray(ann.onset, dtype=float)[idxs]
    times = onsets.copy()

    if ann.orig_time is not None and absolute_start_datetime is not None:
        base_ann = pd.Timestamp(ann.orig_time)
        base_main = pd.Timestamp(absolute_start_datetime)
        if base_main.tz is not None:
            if base_ann.tz is None:
                base_ann = base_ann.tz_localize(base_main.tz)
            else:
                base_ann = base_ann.tz_convert(base_main.tz)
        else:
            base_ann = base_ann.tz_localize(None)
            base_main = base_main.tz_localize(None) if base_main.tz is not None else base_main
        abs_onsets = base_ann + pd.to_timedelta(times, unit='s')
        times = (abs_onsets - base_main).total_seconds().to_numpy()

    df = pd.DataFrame({'time': times, 'stage': codes}).sort_values('time').reset_index(drop=True)

    # Fill leading U
    out = []
    if len(df) and df['time'].iloc[0] > 0.0:
        t = 0.0
        while t < float(df['time'].iloc[0]) - 1e-6:
            out.append({'time': t, 'stage': 'U'})
            t += float(epoch_seconds)
    out_df = pd.concat([pd.DataFrame(out), df], ignore_index=True) if out else df

    # Fill trailing U
    if len(out_df):
        last = float(out_df['time'].iloc[-1])
        t = last + float(epoch_seconds)
        # Limiter la durée maximale pour éviter les boucles infinies (max 24h)
        max_duration = min(float(recording_duration_s), 24 * 3600)
        while t < max_duration - 1e-6:
            out_df.loc[len(out_df)] = {'time': t, 'stage': 'U'}
            t += float(epoch_seconds)
            # Protection contre les boucles infinies
            if t > last + 100 * epoch_seconds:
                break

    return out_df
