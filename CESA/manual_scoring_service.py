"""Manual sleep-scoring service with a strict time/stage contract."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from CESA.scoring_io import import_edf_hypnogram, import_excel_scoring


@dataclass
class ManualScoringResult:
    """Normalized manual scoring output."""

    df: pd.DataFrame
    epoch_seconds: float


class ManualScoringService:
    """Centralized import/normalize/validate utilities for manual scoring."""

    STAGE_MAP = {
        "w": "W",
        "wake": "W",
        "eveil": "W",
        "eveil": "W",
        "e\u0301veil": "W",
        "veille": "W",
        "n1": "N1",
        "n2": "N2",
        "n3": "N3",
        "n4": "N3",
        "sws": "N3",
        "slow wave": "N3",
        "r": "R",
        "rem": "R",
        "paradoxal": "R",
        "u": "U",
        "unk": "U",
        "unknown": "U",
        "artefact": "U",
        "artifact": "U",
        "movement": "U",
        "mt": "U",
        "mvt": "U",
    }
    ALLOWED_STAGES = {"W", "N1", "N2", "N3", "R", "U"}

    @classmethod
    def import_excel_path(
        cls,
        file_path: str,
        *,
        absolute_start_datetime: Optional[pd.Timestamp],
        recording_duration_s: float,
        default_epoch_seconds: float = 30.0,
    ) -> ManualScoringResult:
        raw = cls._read_table(file_path)
        stage_col, time_col = cls._detect_columns(raw)
        work = raw[[stage_col, time_col]].copy()
        work.columns = ["stage", "time_like"]
        work = work.dropna(subset=["stage", "time_like"]).reset_index(drop=True)

        stage_norm = work["stage"].map(cls._normalize_stage)
        stage_norm = stage_norm.fillna("U")

        as_datetime = pd.to_datetime(work["time_like"], errors="coerce")
        datetime_ratio = float(as_datetime.notna().mean()) if len(as_datetime) else 0.0

        if datetime_ratio >= 0.7:
            tmp = pd.DataFrame({"stage": stage_norm, "datetime": as_datetime}).dropna(subset=["datetime"])
            if len(tmp) == 0:
                raise ValueError("Aucune date/heure valide dans le fichier de scoring.")
            df_time = import_excel_scoring(
                tmp,
                absolute_start_datetime=absolute_start_datetime,
                epoch_seconds=float(default_epoch_seconds),
            )
            out = pd.DataFrame({"time": df_time["time"].to_numpy(float), "stage": tmp["stage"].to_numpy(str)})
        else:
            times = pd.to_numeric(work["time_like"], errors="coerce")
            out = pd.DataFrame({"time": times, "stage": stage_norm}).dropna(subset=["time"])

        out = cls.validate(out)
        epoch = cls.infer_epoch_seconds(out, default=float(default_epoch_seconds))
        out = cls.fill_undefined(
            out,
            recording_duration_s=float(recording_duration_s),
            epoch_seconds=float(epoch),
        )
        return ManualScoringResult(df=out, epoch_seconds=float(epoch))

    @classmethod
    def import_edf_path(
        cls,
        file_path: str,
        *,
        recording_duration_s: float,
        epoch_seconds: float,
        absolute_start_datetime: Optional[pd.Timestamp],
    ) -> ManualScoringResult:
        df = import_edf_hypnogram(
            file_path,
            recording_duration_s=float(recording_duration_s),
            epoch_seconds=float(epoch_seconds),
            absolute_start_datetime=absolute_start_datetime,
        )
        out = cls.validate(df)
        epoch = cls.infer_epoch_seconds(out, default=float(epoch_seconds))
        out = cls.fill_undefined(
            out,
            recording_duration_s=float(recording_duration_s),
            epoch_seconds=float(epoch),
        )
        return ManualScoringResult(df=out, epoch_seconds=float(epoch))

    @classmethod
    def validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Le scoring doit etre un DataFrame pandas.")
        if "time" not in df.columns or "stage" not in df.columns:
            raise ValueError("Le scoring doit contenir les colonnes 'time' et 'stage'.")

        out = df[["time", "stage"]].copy()
        out["time"] = pd.to_numeric(out["time"], errors="coerce")
        out["stage"] = out["stage"].map(cls._normalize_stage).fillna("U")
        out = out.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="last")
        out = out.reset_index(drop=True)
        return out

    @classmethod
    def fill_undefined(cls, df: pd.DataFrame, *, recording_duration_s: float, epoch_seconds: float) -> pd.DataFrame:
        out = cls.validate(df)
        if len(out) == 0:
            return out
        epoch = max(1.0, float(epoch_seconds))
        max_dur = min(float(recording_duration_s), 24.0 * 3600.0)
        if max_dur <= 0:
            return out

        first_t = float(out["time"].iloc[0])
        last_t = float(out["time"].iloc[-1])
        add_rows: list[dict[str, object]] = []

        if first_t > 0.0:
            t = 0.0
            while t < first_t - 1e-6:
                add_rows.append({"time": t, "stage": "U"})
                t += epoch

        if last_t + epoch < max_dur - 1e-6:
            t = last_t + epoch
            while t < max_dur - 1e-6:
                add_rows.append({"time": t, "stage": "U"})
                t += epoch

        if add_rows:
            out = pd.concat([out, pd.DataFrame(add_rows)], ignore_index=True)
            out = out.sort_values("time").drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
        return out

    @staticmethod
    def infer_epoch_seconds(df: pd.DataFrame, *, default: float = 30.0) -> float:
        try:
            times = pd.to_numeric(df["time"], errors="coerce").dropna().to_numpy(dtype=float)
            if len(times) < 2:
                return float(default)
            diffs = np.diff(np.sort(times))
            diffs = diffs[diffs > 0]
            if len(diffs) == 0:
                return float(default)
            med = float(np.median(diffs))
            if 5.0 <= med <= 120.0:
                return med
        except Exception as exc:
            logging.debug("infer_epoch_seconds failed: %s", exc)
        return float(default)

    @classmethod
    def _detect_columns(cls, df: pd.DataFrame) -> tuple[str, str]:
        cols = [str(c).strip() for c in df.columns]
        if len(cols) < 2:
            raise ValueError("Le fichier de scoring doit contenir au moins deux colonnes.")
        stage_col: Optional[str] = None
        time_col: Optional[str] = None

        for col in cols:
            low = col.lower()
            if stage_col is None and any(k in low for k in ("stage", "stade", "sleep", "sommeil", "state", "score")):
                stage_col = col
            if time_col is None and any(k in low for k in ("time", "temps", "datetime", "date", "heure", "epoch", "epoque", "start", "debut")):
                time_col = col

        if stage_col is None:
            stage_col = cols[0]
        if time_col is None:
            time_col = cols[1] if len(cols) > 1 else cols[0]
        if stage_col == time_col:
            time_col = cols[1] if len(cols) > 1 else cols[0]
        return stage_col, time_col

    @classmethod
    def _normalize_stage(cls, value: object) -> str:
        raw = str(value).strip().upper()
        if raw in cls.ALLOWED_STAGES:
            return raw
        low = str(value).strip().lower()
        mapped = cls.STAGE_MAP.get(low)
        return mapped if mapped is not None else "U"

    @staticmethod
    def _read_table(file_path: str) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {file_path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(str(path), encoding="utf-8")
        if suffix in {".xls", ".xlsx"}:
            return pd.read_excel(str(path))
        raise ValueError("Format non supporte. Utilisez CSV, XLS ou XLSX.")
