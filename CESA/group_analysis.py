"""
Group-level analysis utilities for multiscale entropy (MSE) and renormalised entropy.

Le module regroupe :
- le chargement d'un design expérimental (Sujet / Condition / chemins EDF / scoring),
- l'extraction optionnelle par stade de sommeil,
- le calcul des profils MSE ou d'entropie renormée pour chaque combinaison,
- les tests statistiques appariés (Wilcoxon, permutation robuste, bootstrap, Z robuste),
- et des helpers pour tracer les profils façon « spaghetti ».
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import logging
import math
import os
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats

try:  # pragma: no cover - dépendance optionnelle lors des tests
    import mne
except Exception:  # pragma: no cover
    mne = None  # type: ignore

from .entropy import (
    MultiscaleEntropyConfig,
    RenormalizedEntropyConfig,
    compute_multiscale_entropy,
    compute_renormalized_entropy,
)
from .cardiac_hrv import HRVConfig, HRVComputationError, compute_epoch_hrv, _normalize_stage_label
from .scoring_io import import_excel_scoring, import_edf_hypnogram
from .advanced_spaghetti_plots import (
    permutation_test_median_diff,
    bootstrap_ci_median_diff,
    robust_z_intrasubject,
)
from core.raw_loader import open_raw_file

LOGGER = logging.getLogger(__name__)
DEFAULT_STAGES: Tuple[str, ...] = ("ALL", "W", "N1", "N2", "N3", "R")
_STAGE_ALIASES: Dict[str, str] = {
    "": "ALL",
    "ALL": "ALL",
    "*": "ALL",
    "TOUT": "ALL",
    "W": "W",
    "WAKE": "W",
    "ÉVEIL": "W",
    "EVEIL": "W",
    "AWAKE": "W",
    "N1": "N1",
    "S1": "N1",
    "STAGE1": "N1",
    "STAGE 1": "N1",
    "N2": "N2",
    "S2": "N2",
    "STAGE2": "N2",
    "STAGE 2": "N2",
    "N3": "N3",
    "S3": "N3",
    "S4": "N3",
    "STAGE3": "N3",
    "STAGE 3": "N3",
    "STAGE4": "N3",
    "STAGE 4": "N3",
    "R": "R",
    "REM": "R",
    "PARADOXAL": "R",
}


@dataclass
class GroupDesignEntry:
    """Représente une ligne du design expérimental."""

    subject: str
    condition: str
    edf_path: Path
    scoring_path: Optional[Path] = None
    stages: Optional[Tuple[str, ...]] = None

    def resolved_stages(self, default: Sequence[str]) -> Tuple[str, ...]:
        if self.stages:
            return tuple(self.stages)
        return tuple(default)


@dataclass
class GroupAnalysisConfig:
    """Configuration principale de l'analyse de groupe."""

    metric: str = "mse"  # mse | renorm
    before_label: str = "AVANT"
    after_label: str = "APRÈS"
    display_before_label: str = "AVANT"
    display_after_label: str = "APRÈS"
    display_channels: Optional[str] = None
    stages: Sequence[str] = field(default_factory=lambda: DEFAULT_STAGES)
    epoch_seconds: float = 30.0
    channel_names: Optional[Sequence[str]] = None
    mse_config: MultiscaleEntropyConfig = field(default_factory=MultiscaleEntropyConfig)
    renorm_config: RenormalizedEntropyConfig = field(default_factory=RenormalizedEntropyConfig)
    hrv_config: HRVConfig = field(default_factory=HRVConfig)
    max_workers: int = 0  # 0 => auto (calqué sur le nombre de cœurs disponibles)
    cpu_proportion: float = 1.0  # Proportion de CPU à utiliser (0.0-1.0), ignoré si max_workers > 0

    def normalised_stage_list(self) -> List[str]:
        stages: List[str] = []
        for stage in self.stages:
            norm = normalise_stage(stage)
            if norm not in stages:
                stages.append(norm)
        return stages


@dataclass
class StatsConfig:
    """Options pour les tests statistiques."""

    run_wilcoxon: bool = True
    run_permutation: bool = True
    run_bootstrap: bool = False
    run_robust_z: bool = False
    n_permutations: int = 5000
    apply_bh: bool = True
    alpha: float = 0.05


@dataclass
class GroupAnalysisResult:
    """Conteneur des résultats."""

    profiles: pd.DataFrame  # colonnes: subject, condition, stage, tau, value
    subject_summary: pd.DataFrame  # colonnes: subject, condition, stage, auc, mean
    stats: Optional[pd.DataFrame]
    config: GroupAnalysisConfig

    def available_stages(self) -> List[str]:
        if self.profiles.empty:
            return []
        return sorted(self.profiles["stage"].unique().tolist())

    def tau_values(self) -> List[float]:
        if self.profiles.empty:
            return []
        # Exclure tau=0 qui est le CI, pas une vraie échelle d'entropie
        taus = sorted([tau for tau in self.profiles["tau"].unique().tolist() if tau > 0])
        return taus


class GroupAnalysisError(RuntimeError):
    """Exception spécifique à l'analyse de groupe."""


@dataclass
class GroupHRVResult:
    """Résultats HRV par époque et par sujet/condition."""

    epochs: pd.DataFrame  # colonnes: subject, condition, stage, segment_start_s/stop_s, rmssd, lf, hf, lf_hf, n_peaks, n_rr, channel, rr_reject_pct, rr_correction_events, rr_median, peak_rate_bpm, hr_median_bpm, peak_detection_method, rr_cleaning_method, hrv_quality, hrv_quality_reason
    summary: pd.DataFrame  # colonnes: subject, condition, stage, rmssd, lf, hf, lf_hf, rr_correction_events, peak_rate_bpm, hr_median_bpm, peak_detection_method, rr_cleaning_method, hrv_quality, hrv_quality_reason, n_segments, duration_s
    config: GroupAnalysisConfig


# ============================================================================
# Chargement du design
# ============================================================================


def load_design_csv(path: str | Path) -> List[GroupDesignEntry]:
    """Charge un fichier CSV décrivant le design expérimental."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise GroupAnalysisError(f"Fichier design introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"subject", "condition", "edf_path"}
    missing = required.difference({c.lower() for c in df.columns})
    if missing:
        raise GroupAnalysisError(
            f"Colonnes obligatoires manquantes ({', '.join(sorted(required))}) dans {csv_path}"
        )

    def _get_col(name: str) -> Optional[str]:
        for col in df.columns:
            if col.lower() == name:
                return col
        return None

    col_subject = _get_col("subject")
    col_condition = _get_col("condition")
    col_edf = _get_col("edf_path")
    col_scoring = _get_col("scoring_path")
    col_stage = _get_col("stage")

    entries: List[GroupDesignEntry] = []
    for _, row in df.iterrows():
        subject = str(row[col_subject]).strip()
        condition = str(row[col_condition]).strip()
        edf_path = Path(str(row[col_edf]).strip())
        if not subject or not condition or not edf_path:
            continue

        scoring_path = None
        if col_scoring and not pd.isna(row[col_scoring]):
            scoring_path = Path(str(row[col_scoring]).strip())

        stages: Optional[Tuple[str, ...]] = None
        if col_stage and not pd.isna(row[col_stage]):
            stages = tuple(parse_stage_field(str(row[col_stage])))

        entries.append(
            GroupDesignEntry(
                subject=subject,
                condition=condition,
                edf_path=edf_path,
                scoring_path=scoring_path,
                stages=stages,
            )
        )

    if not entries:
        raise GroupAnalysisError("Aucune entrée valide détectée dans le design")

    return entries


def parse_stage_field(value: str) -> List[str]:
    """Décode un champ Stage libre (ex: \"W,N2\" ou \"*\" pour tous)."""

    if not value:
        return ["ALL"]
    parts = [normalise_stage(part.strip()) for part in value.replace(";", ",").split(",")]
    unique = []
    for stage in parts:
        if stage not in unique:
            unique.append(stage)
    return unique or ["ALL"]


# ============================================================================
# Calcul des profils
# ============================================================================


def compute_group_profiles(
    entries: Sequence[GroupDesignEntry],
    config: GroupAnalysisConfig,
    *,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> GroupAnalysisResult:
    """Calcule les profils (MSE ou entropie renormée) pour un design donné."""

    if mne is None:
        raise GroupAnalysisError("La dépendance mne est requise pour charger les fichiers EDF.")

    stage_list = tuple(config.normalised_stage_list())
    records: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    total_entries = len(entries)
    if total_entries == 0:
        raise GroupAnalysisError("Aucune entrée n'a été fournie.")

    cpu_count = os.cpu_count() or 1
    
    # Calcul du nombre de workers selon la configuration
    if config.max_workers > 0:
        # Nombre absolu spécifié
        desired_workers = config.max_workers
    else:
        # Utiliser la proportion CPU
        cpu_proportion = max(0.0, min(1.0, config.cpu_proportion))  # Clamp entre 0 et 1
        desired_workers = max(1, int(round(cpu_count * cpu_proportion)))
    
    workers = max(1, min(desired_workers, total_entries))
    
    # Afficher les informations de configuration
    if config.max_workers > 0:
        print(
            f"⚙️ CHECKPOINT GROUP: utilisation de {workers} worker(s) (CPU dispo: {cpu_count}, "
            f"configuré: {config.max_workers})",
            flush=True,
        )
    else:
        print(
            f"⚙️ CHECKPOINT GROUP: utilisation de {workers} worker(s) (CPU dispo: {cpu_count}, "
            f"proportion: {config.cpu_proportion*100:.0f}%)",
            flush=True,
        )

    if workers == 1:
        for idx, entry in enumerate(entries, start=1):
            _notify(
                progress_cb,
                f"Sujet {idx}/{total_entries} – {entry.subject} / {entry.condition} (en cours)",
                max(0.0, (idx - 1) / total_entries),
            )
            entry_records, entry_summary = _process_entry(entry, config, stage_list, idx, total_entries)
            records.extend(entry_records)
            summary_rows.extend(entry_summary)
            _notify(
                progress_cb,
                f"Sujet {idx}/{total_entries} – {entry.subject} / {entry.condition}",
                idx / total_entries,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_entry, entry, config, stage_list, idx, total_entries): idx
                for idx, entry in enumerate(entries, start=1)
            }
            completed = 0
            for future in as_completed(futures):
                entry_records, entry_summary = future.result()
                records.extend(entry_records)
                summary_rows.extend(entry_summary)
                completed += 1
                idx = futures[future]
                entry = entries[idx - 1]
                _notify(
                    progress_cb,
                    f"Sujet {completed}/{total_entries} – {entry.subject} / {entry.condition}",
                    completed / total_entries,
                )

    profiles_df = pd.DataFrame.from_records(records)
    summary_df = pd.DataFrame.from_records(summary_rows)

    if profiles_df.empty:
        raise GroupAnalysisError("Aucun profil n'a pu être calculé (vérifiez les fichiers et le scoring).")

    return GroupAnalysisResult(
        profiles=profiles_df,
        subject_summary=summary_df,
        stats=None,
        config=config,
    )


def compute_group_hrv(
    entries: Sequence[GroupDesignEntry],
    config: GroupAnalysisConfig,
    *,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> GroupHRVResult:
    """Calcule HRV sur les époques d'éveil (W) pour chaque entrée du design."""

    if mne is None:
        raise GroupAnalysisError("La dépendance mne est requise pour charger les fichiers EDF.")

    if not entries:
        raise GroupAnalysisError("Aucune entrée n'a été fournie.")

    records: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    total_entries = len(entries)

    for idx, entry in enumerate(entries, start=1):
        _notify(progress_cb, f"HRV sujet {idx}/{total_entries} – {entry.subject}", (idx - 1) / total_entries)
        LOGGER.info("[HRV] Sujet=%s Condition=%s (%d/%d)", entry.subject, entry.condition, idx, total_entries)
        raw = None
        try:
            if not entry.edf_path.exists():
                LOGGER.warning("EDF introuvable: %s", entry.edf_path)
                continue
            raw = open_raw_file(str(entry.edf_path), preload=False, verbose="ERROR")  # type: ignore[arg-type]
            scoring_df = _load_scoring(entry, raw, config.epoch_seconds)
            if scoring_df is None:
                LOGGER.warning("Pas de scoring pour %s", entry.edf_path)
                continue
            try:
                epoch_records = compute_epoch_hrv(
                    raw,
                    scoring_df,
                    epoch_seconds=config.epoch_seconds,
                    channel_names=config.channel_names,
                    config=config.hrv_config,
                )
            except HRVComputationError as exc:
                LOGGER.warning("HRV échoué pour %s: %s", entry.edf_path, exc)
                # Continuer aux sujets suivants, mais on notera si rien n'est agrégé
                continue

            for rec in epoch_records:
                rec["subject"] = entry.subject
                rec["condition"] = entry.condition
                records.append(rec)
            LOGGER.info(
                "[HRV] %s/%s: %d segments, canal=%s, rmssd_med=%.3f",
                entry.subject,
                entry.condition,
                len(epoch_records),
                epoch_records[0].get("channel") if epoch_records else "n/a",
                float(np.median([r.get("rmssd", np.nan) for r in epoch_records])) if epoch_records else float("nan"),
            )

        finally:
            if raw is not None:
                raw.close()

    epochs_df = pd.DataFrame.from_records(records)
    if epochs_df.empty:
        raise GroupAnalysisError(
            "Aucun segment HRV n'a été calculé. "
            "Conseil: réduire la durée min des segments ou cocher \"segments courts\" dans Paramètres HRV, "
            "et vérifier les stades sélectionnés (ex: REM,R ou W)."
        )

    # Convertir en numérique et filtrer par stades demandés (au cas où)
    numeric_cols = ["rmssd", "lf", "hf", "lf_hf", "duration_s", "rr_reject_pct", "rr_correction_events", "rr_median", "peak_rate_bpm", "hr_median_bpm", "n_rr_raw", "n_rr"]
    for col in numeric_cols:
        if col in epochs_df.columns:
            epochs_df[col] = pd.to_numeric(epochs_df[col], errors="coerce")
    # Filtrer sur les stades de la config HRV
    stage_norm = {_normalize_stage_label(s) for s in config.hrv_config.stage_filter}
    if stage_norm:
        epochs_df = epochs_df[epochs_df["stage"].isin(stage_norm)]
    if epochs_df.empty:
        raise GroupAnalysisError("Aucun segment HRV après filtrage par stades sélectionnés.")

    # Agrégation par sujet/condition/stage
    def _agg(df: pd.DataFrame, col: str) -> float:
        if df[col].dropna().empty:
            return float("nan")
        return float(df[col].median())

    def _mode_or_default(df: pd.DataFrame, col: str, default: str) -> str:
        if col not in df.columns:
            return default
        series = df[col].dropna()
        if series.empty:
            return default
        return str(series.mode().iloc[0])

    def _worst_quality(df: pd.DataFrame) -> str:
        if "hrv_quality" not in df.columns:
            return "unknown"
        ranking = {"good": 0, "acceptable": 1, "poor": 2, "unknown": -1}
        qualities = [str(v) for v in df["hrv_quality"].dropna().tolist()]
        if not qualities:
            return "unknown"
        return max(qualities, key=lambda value: ranking.get(value, -1))

    def _worst_quality_reason(df: pd.DataFrame) -> str:
        if "hrv_quality_reason" not in df.columns:
            return "unknown"
        ranking = {
            "ok": 0,
            "unknown": 0,
            "hr_median_out_of_range": 1,
            "peak_rate_out_of_range": 1,
            "peak_hr_mismatch": 1,
            "high_rmssd_review": 1,
            "extreme_lf_hf": 1,
            "moderate_rr_reject_pct": 2,
            "insufficient_rr": 2,
            "high_rr_reject_pct": 3,
        }
        reasons = [str(v) for v in df["hrv_quality_reason"].dropna().tolist()]
        if not reasons:
            return "unknown"
        return max(reasons, key=lambda value: ranking.get(value, 0))

    grouped = []
    for (subj, cond, stage), grp in epochs_df.groupby(["subject", "condition", "stage"]):
        method_value = _mode_or_default(grp, "rr_cleaning_method", str(getattr(config.hrv_config, "rr_cleaning_method", "simple")))
        peak_method_value = _mode_or_default(grp, "peak_detection_method", str(getattr(config.hrv_config, "peak_detection_method", "simple")))
        grouped.append(
            {
                "subject": subj,
                "condition": cond,
                "stage": stage,
                "rmssd": _agg(grp, "rmssd"),
                "lf": _agg(grp, "lf"),
                "hf": _agg(grp, "hf"),
                "lf_hf": _agg(grp, "lf_hf"),
                "rr_reject_pct": _agg(grp, "rr_reject_pct"),
                "rr_correction_events": _agg(grp, "rr_correction_events"),
                "rr_median": _agg(grp, "rr_median"),
                "peak_rate_bpm": _agg(grp, "peak_rate_bpm"),
                "hr_median_bpm": _agg(grp, "hr_median_bpm"),
                "freq_domain_ok": bool(grp["freq_domain_ok"].all()) if "freq_domain_ok" in grp.columns else True,
                "peak_detection_method": peak_method_value,
                "rr_cleaning_method": method_value,
                "hrv_quality": _worst_quality(grp),
                "hrv_quality_reason": _worst_quality_reason(grp),
                "n_segments": int(len(grp)),
                "duration_s": float(grp["duration_s"].sum()) if "duration_s" in grp.columns else float("nan"),
            }
        )
    summary_df = pd.DataFrame(grouped)
    LOGGER.info(
        "[HRV] Terminé: %d sujets, %d conditions, %d époques, %d lignes résumé",
        epochs_df["subject"].nunique(),
        epochs_df["condition"].nunique(),
        len(epochs_df),
        len(summary_df),
    )
    _notify(progress_cb, "HRV terminé", 1.0)

    return GroupHRVResult(epochs=epochs_df, summary=summary_df, config=config)


def run_statistical_tests(
    result: GroupAnalysisResult,
    stats_cfg: StatsConfig,
) -> pd.DataFrame:
    """Exécute les tests appariés sur les profils déjà calculés."""

    df = result.profiles.copy()
    cfg = result.config

    print(
        f"📊 CHECKPOINT STATS: démarrage – {len(result.available_stages())} stades, "
        f"{df['subject'].nunique()} sujets uniques, "
        f"tests activés: "
        f"{'W' if stats_cfg.run_wilcoxon else ''}"
        f"{'P' if stats_cfg.run_permutation else ''}"
        f"{'B' if stats_cfg.run_bootstrap else ''}"
        f"{'Z' if stats_cfg.run_robust_z else ''}",
        flush=True,
    )

    if cfg.display_before_label and normalise_condition_label(cfg.display_before_label) != normalise_condition_label(cfg.before_label):
        print(
            f"[STATS] Label avant saisi « {cfg.display_before_label} » → colonne réelle « {cfg.before_label} ».",
            flush=True,
        )
    if cfg.display_after_label and normalise_condition_label(cfg.display_after_label) != normalise_condition_label(cfg.after_label):
        print(
            f"[STATS] Label après saisi « {cfg.display_after_label} » → colonne réelle « {cfg.after_label} ».",
            flush=True,
        )

    before_label = cfg.before_label
    after_label = cfg.after_label

    records: List[Dict[str, object]] = []

    for stage in result.available_stages():
        stage_df = df[df["stage"] == stage]
        if stage_df.empty:
            print(f"[STATS] Stade {stage}: aucune donnée, passage.", flush=True)
            continue
        print(
            f"[STATS] Stade {stage}: {stage_df['subject'].nunique()} sujets, "
            f"{len(stage_df['tau'].unique())} échelles τ.",
            flush=True,
        )
        # Exclure tau=0 (CI) des tests statistiques - ce n'est pas une échelle d'entropie
        valid_taus = [tau for tau in sorted(stage_df["tau"].unique()) if tau > 0]
        for tau in valid_taus:
            subset = stage_df[stage_df["tau"] == tau]
            pivot = subset.pivot_table(index="subject", columns="condition", values="value")
            before = _match_condition_column(pivot.columns, before_label)
            after = _match_condition_column(pivot.columns, after_label)

            if not before or not after or before == after:
                if len(pivot.columns) >= 2:
                    sorted_cols = sorted(pivot.columns)
                    auto_before, auto_after = sorted_cols[0], sorted_cols[1]
                    print(
                        f"[STATS] Colonnes {before_label}/{after_label} absentes pour {stage} τ={tau}; "
                        f"auto={auto_before}/{auto_after}",
                        flush=True,
                    )
                    before = auto_before
                    after = auto_after
                else:
                    print(
                        f"[STATS] Stade {stage} τ={tau}: moins de 2 conditions présentes ({list(pivot.columns)}), "
                        "tests ignorés.",
                        flush=True,
                    )
                    continue

            paired = pivot[[before, after]].dropna()
            if paired.empty:
                print(
                    f"[STATS] Stade {stage} τ={tau}: aucune paire complète {before}/{after}.",
                    flush=True,
                )
                continue
            x_before = paired[before].to_numpy(dtype=float)
            x_after = paired[after].to_numpy(dtype=float)
            diff = x_after - x_before

            tests_done = 0
            if stats_cfg.run_wilcoxon:
                try:
                    from scipy.stats import wilcoxon  # type: ignore

                    if diff.size >= 1:
                        stat, p_value = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", mode="auto")
                    else:
                        stat, p_value = (np.nan, np.nan)
                except Exception:
                    stat, p_value = (np.nan, np.nan)
                records.append(
                    {
                        "stage": stage,
                        "tau": float(tau),
                        "test": "wilcoxon",
                        "statistic": float(stat) if not np.isnan(stat) else np.nan,
                        "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
                        "n_subjects": int(diff.size),
                    }
                )
                tests_done += 1

            if stats_cfg.run_permutation:
                perm = permutation_test_median_diff(
                    x_before,
                    x_after,
                    n_perm=stats_cfg.n_permutations,
                )
                records.append(
                    {
                        "stage": stage,
                        "tau": float(tau),
                        "test": "permutation",
                        "statistic": perm.get("D_obs"),
                        "p_value": perm.get("p_value"),
                        "decision": perm.get("decision"),
                        "n_subjects": int(diff.size),
                    }
                )
                tests_done += 1

            if stats_cfg.run_bootstrap:
                boot = bootstrap_ci_median_diff(x_before, x_after)
                records.append(
                    {
                        "stage": stage,
                        "tau": float(tau),
                        "test": "bootstrap",
                        "statistic": boot.get("D_obs"),
                        "p_value": np.nan,
                        "ci_low": boot.get("CI_low"),
                        "ci_high": boot.get("CI_high"),
                        "decision": boot.get("decision"),
                        "n_subjects": int(diff.size),
                    }
                )
                tests_done += 1

            if stats_cfg.run_robust_z:
                rz = robust_z_intrasubject(x_before, x_after)
                records.append(
                    {
                        "stage": stage,
                        "tau": float(tau),
                        "test": "robust_z",
                        "statistic": rz.get("Z"),
                        "p_value": np.nan,
                        "decision": rz.get("decision"),
                        "n_subjects": int(diff.size),
                    }
                )
                tests_done += 1

            print(
                f"[STATS] Stade {stage} τ={tau}: paires={diff.size}, tests effectués={tests_done}",
                flush=True,
            )

    stats_df = pd.DataFrame.from_records(records)
    if stats_df.empty:
        print("⚠️ CHECKPOINT STATS: aucun test généré (voir messages ci-dessus).", flush=True)
        result.stats = pd.DataFrame()
        return result.stats

    if stats_cfg.apply_bh and "p_value" in stats_df.columns:
        for test_name, group in stats_df.groupby("test"):
            mask = group["p_value"].notna()
            if not mask.any():
                continue
            corrected = benjamini_hochberg(group.loc[mask, "p_value"].to_numpy(dtype=float), alpha=stats_cfg.alpha)
            stats_df.loc[group.index[mask], "p_adj"] = corrected

    result.stats = stats_df
    print(f"📁 CHECKPOINT STATS: terminé – {len(stats_df)} lignes exportables.", flush=True)
    return stats_df


# ============================================================================
# Export & Visualisation
# ============================================================================


def export_profiles_to_csv(result: GroupAnalysisResult, path: str | Path) -> None:
    """Sauvegarde les profils détaillés dans un CSV."""

    df = result.profiles.sort_values(["stage", "subject", "tau"])
    _write_csv_with_metadata(path, df, result.config)


def export_stats_to_csv(result: GroupAnalysisResult, path: str | Path) -> None:
    """Sauvegarde la table de tests statistiques."""

    if result.stats is None or result.stats.empty:
        raise GroupAnalysisError("Aucune statistique n'est disponible pour l'export.")
    df = result.stats.sort_values(["test", "stage", "tau"])
    _write_csv_with_metadata(path, df, result.config)


def export_hrv_to_csv(hrv_result: GroupHRVResult, path: str | Path, summary_only: bool = False) -> None:
    """Sauvegarde les HRV (époques ou résumé) dans un CSV."""

    df = hrv_result.summary if summary_only else hrv_result.epochs
    df = df.copy()
    df = df.sort_values(
        [
            "subject",
            "condition",
            "stage",
            "segment_start_s" if "segment_start_s" in df.columns else ("epoch_start_s" if "epoch_start_s" in df.columns else "n_segments"),
        ]
    )
    metadata = [
        f"Metric: HRV",
        f"Label avant: {hrv_result.config.display_before_label or hrv_result.config.before_label}",
        f"Label après: {hrv_result.config.display_after_label or hrv_result.config.after_label}",
        f"Epoch (s): {hrv_result.config.epoch_seconds}",
        f"LF band: {hrv_result.config.hrv_config.lf_band}",
        f"HF band: {hrv_result.config.hrv_config.hf_band}",
        f"Resample fs: {hrv_result.config.hrv_config.resample_fs}",
        f"Peak detection method: {getattr(hrv_result.config.hrv_config, 'peak_detection_method', 'simple')}",
        f"RR cleaning method: {getattr(hrv_result.config.hrv_config, 'rr_cleaning_method', 'simple')}",
        "LF/HF note: métrique secondaire et interprétative, sensible à la stationnarité et à HF faible",
        f"QC columns: peak_rate_bpm, hr_median_bpm, hrv_quality, hrv_quality_reason",
    ]
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        for line in metadata:
            handle.write(f"# {line}\n")
        df.to_csv(handle, index=False)


def compute_hrv_stats(
    hrv_result: GroupHRVResult,
    before_label: Optional[str] = None,
    after_label: Optional[str] = None,
) -> pd.DataFrame:
    """Stats HRV appariées, stratifiées par stade si nécessaire."""

    df = hrv_result.summary.copy()
    if df.empty:
        LOGGER.info("[HRV_STATS] Pas de données (summary vide)")
        return pd.DataFrame()

    cfg = hrv_result.config
    before = before_label or cfg.before_label or cfg.display_before_label or "AVANT"
    after = after_label or cfg.after_label or cfg.display_after_label or "APRÈS"
    before_norm = normalise_condition_label(before)
    after_norm = normalise_condition_label(after)

    def _norm_cond(x: str) -> str:
        return normalise_condition_label(x)

    def _subject_base(name: str) -> str:
        parts = str(name).split("_")
        return parts[0] if parts else str(name)

    df = df.copy()
    df["condition_norm"] = df["condition"].map(_norm_cond)
    df["subject_base"] = df["subject"].map(_subject_base)
    stages = sorted(df["stage"].dropna().astype(str).unique().tolist()) if "stage" in df.columns else ["ALL"]
    metrics = ["rmssd", "lf", "hf", "lf_hf"]
    records: List[Dict[str, object]] = []

    def _consensus_from_decisions(decisions: Sequence[str]) -> Tuple[str, bool]:
        num_up = sum(dec == "augmentation" for dec in decisions)
        num_down = sum(dec == "diminution" for dec in decisions)
        if num_up >= 2:
            return "augmentation", True
        if num_down >= 2:
            return "diminution", True
        return "stagnation", False

    for stage in stages:
        stage_df = df[df["stage"] == stage].copy() if "stage" in df.columns else df.copy()
        if stage_df.empty:
            continue
        stage_df = stage_df.sort_values(["subject_base", "condition_norm"])
        stage_df = stage_df.groupby(["subject_base", "condition_norm"], as_index=False).last()

        for metric in metrics:
            metric_df = stage_df[["subject_base", "condition_norm", metric]].dropna()
            if metric_df.empty:
                LOGGER.info("[HRV_STATS] %s/%s: aucune valeur non-NaN", stage, metric)
                continue
            pivot = metric_df.pivot_table(index="subject_base", columns="condition_norm", values=metric)
            available_cols = list(pivot.columns)
            LOGGER.info(
                "[HRV_STATS] stage=%s metric=%s: conditions dispo=%s before=%s after=%s",
                stage,
                metric,
                available_cols,
                before,
                after,
            )

            b_col = _match_condition_column(pivot.columns, before_norm)
            a_col = _match_condition_column(pivot.columns, after_norm)
            if (not b_col or not a_col or b_col == a_col) and len(pivot.columns) >= 2:
                counts = pivot.count().sort_values(ascending=False)
                top_cols = list(counts.index[:2])
                if len(top_cols) == 2:
                    b_col, a_col = top_cols[0], top_cols[1]
            if (not b_col or not a_col or b_col == a_col) and len(pivot.columns) >= 2:
                cols = sorted(pivot.columns)
                b_col, a_col = cols[0], cols[1]
            if not b_col or not a_col or b_col == a_col:
                LOGGER.info("[HRV_STATS] stage=%s metric=%s: conditions insuffisantes (%s, %s)", stage, metric, b_col, a_col)
                continue

            pairs = pivot[[b_col, a_col]]
            pre_drop = len(pairs)
            pairs = pairs.dropna()
            LOGGER.info(
                "[HRV_STATS] stage=%s metric=%s: colonnes choisies=%s/%s, n_avant_drop=%d, n_paires=%d",
                stage,
                metric,
                b_col,
                a_col,
                pre_drop,
                len(pairs),
            )
            if pairs.empty:
                LOGGER.info("[HRV_STATS] stage=%s metric=%s: aucune paire disponible", stage, metric)
                continue
            x_before = pairs[b_col].to_numpy(dtype=float)
            x_after = pairs[a_col].to_numpy(dtype=float)
            try:
                stat, p = stats.wilcoxon(x_before, x_after, zero_method="wilcox", alternative="two-sided", mode="auto")
            except Exception:
                stat, p = np.nan, np.nan
            perm = permutation_test_median_diff(x_before, x_after)
            boot = bootstrap_ci_median_diff(x_before, x_after)
            rz = robust_z_intrasubject(x_before, x_after)
            consensus_decision, consensus_star = _consensus_from_decisions(
                [
                    str(perm.get("decision", "stagnation")),
                    str(boot.get("decision", "stagnation")),
                    str(rz.get("decision", "stagnation")),
                ]
            )
            records.append(
                {
                    "stage": stage,
                    "metric": metric,
                    "test": "wilcoxon",
                    "before": b_col,
                    "after": a_col,
                    "n_pairs": int(len(pairs)),
                    "statistic": float(stat) if stat is not None else np.nan,
                    "p_value": float(p) if p is not None else np.nan,
                    "perm_statistic": perm.get("D_obs"),
                    "perm_p_value": perm.get("p_value"),
                    "perm_decision": perm.get("decision"),
                    "boot_ci_low": boot.get("CI_low"),
                    "boot_ci_high": boot.get("CI_high"),
                    "boot_decision": boot.get("decision"),
                    "robust_z": rz.get("Z"),
                    "robust_z_decision": rz.get("decision"),
                    "consensus_decision": consensus_decision,
                    "consensus_star": bool(consensus_star),
                }
            )

    out = pd.DataFrame(records)
    if out.empty:
        LOGGER.info("[HRV_STATS] lignes=0")
        return out

    for stage, group in out.groupby("stage"):
        mask = group["p_value"].notna()
        if not mask.any():
            continue
        corrected = benjamini_hochberg(group.loc[mask, "p_value"].to_numpy(dtype=float), alpha=0.05)
        out.loc[group.index[mask], "p_adj"] = corrected

    LOGGER.info("[HRV_STATS] lignes=%d", len(out))
    return out


def save_hrv_plots(hrv_result: GroupHRVResult, base_path: str | Path) -> Tuple[List[Path], pd.DataFrame]:
    """Génère des figures HRV stratifiées par stade si nécessaire."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns  # type: ignore

    cfg = hrv_result.config
    df_summary = hrv_result.summary.copy()
    df_epochs = hrv_result.epochs.copy()
    if df_summary.empty:
        raise GroupAnalysisError("Aucun HRV disponible pour tracer.")

    before = cfg.before_label or cfg.display_before_label or "AVANT"
    after = cfg.after_label or cfg.display_after_label or "APRÈS"
    stages = sorted(df_summary["stage"].dropna().astype(str).unique().tolist()) if "stage" in df_summary.columns else ["ALL"]
    multi_stage = len(stages) > 1

    out_paths: List[Path] = []
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    stats_df = compute_hrv_stats(hrv_result, before_label=before, after_label=after)
    if stats_df.empty:
        LOGGER.info("[HRV_PLOTS] Stats vides (pas de paires exploitables)")

    def _stars(p: float) -> str:
        if np.isnan(p):
            return "ns"
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    def _condition_order(series: pd.Series) -> List[str]:
        conditions = series.dropna().astype(str).unique().tolist()
        order: List[str] = []
        if before in conditions:
            order.append(before)
        if after in conditions and after != before:
            order.append(after)
        for cond in conditions:
            if cond not in order:
                order.append(cond)
        return order

    def _annotate_stat(ax, metric: str, stage: str) -> None:
        if stats_df.empty:
            return
        row = stats_df[(stats_df["metric"] == metric) & (stats_df["stage"] == stage)].head(1)
        if row.empty:
            return
        p = row["p_value"].iloc[0]
        q = row["p_adj"].iloc[0] if "p_adj" in row.columns else np.nan
        n = int(row["n_pairs"].iloc[0])
        stars = _stars(q if not np.isnan(q) else p)
        if not np.isnan(q):
            label = f"Wilcoxon n={n}, p={p:.3g}, q={q:.3g} ({stars})"
        else:
            label = f"Wilcoxon n={n}, p={p:.3g} ({stars})" if not np.isnan(p) else f"Wilcoxon n={n}, p=NA"
        cons_dec = str(row["consensus_decision"].iloc[0]) if "consensus_decision" in row.columns else "stagnation"
        cons_star = bool(row["consensus_star"].iloc[0]) if "consensus_star" in row.columns else False
        cons_text = "="
        if cons_dec == "augmentation":
            cons_text = "UP"
        elif cons_dec == "diminution":
            cons_text = "DN"
        cons_label = f"Cons: {cons_text}{' *' if cons_star else ''}"
        ax.text(0.5, 0.97, label, ha="center", va="top", transform=ax.transAxes, fontsize=9)
        ax.text(0.5, 0.89, cons_label, ha="center", va="top", transform=ax.transAxes, fontsize=9)

    def _stage_suffix(stage: str) -> str:
        return f"_{stage}" if multi_stage else ""

    def _subject_base(name: str) -> str:
        parts = str(name).split("_")
        return parts[0] if parts else str(name)

    for stage in stages:
        stage_summary = df_summary[df_summary["stage"] == stage].copy() if "stage" in df_summary.columns else df_summary.copy()
        stage_epochs = df_epochs[df_epochs["stage"] == stage].copy() if "stage" in df_epochs.columns else df_epochs.copy()
        suffix = _stage_suffix(stage)
        stage_summary["subject_base"] = stage_summary["subject"].map(_subject_base)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric, title in [
            (axes[0], "rmssd", f"RMSSD (paired){' – ' + stage if multi_stage else ''}"),
            (axes[1], "lf_hf", f"Ratio LF/HF (paired, interprétatif){' – ' + stage if multi_stage else ''}"),
        ]:
            pivot = stage_summary.pivot_table(index="subject_base", columns="condition", values=metric)
            b_col = _match_condition_column(pivot.columns, before)
            a_col = _match_condition_column(pivot.columns, after)
            if (not b_col or not a_col or b_col == a_col) and len(pivot.columns) >= 2:
                counts = pivot.count().sort_values(ascending=False)
                top_cols = list(counts.index[:2])
                if len(top_cols) == 2:
                    b_col, a_col = top_cols[0], top_cols[1]
            if (not b_col or not a_col or b_col == a_col) and len(pivot.columns) >= 2:
                cols = sorted(pivot.columns)
                b_col, a_col = cols[0], cols[1]
            if not b_col or not a_col or b_col == a_col:
                ax.set_title(f"{title}\n(conditions insuffisantes)")
                ax.grid(alpha=0.2)
                continue
            pairs = pivot[[b_col, a_col]].dropna()
            if pairs.empty:
                ax.set_title(f"{title}\n(pas de paires)")
                ax.grid(alpha=0.2)
                continue
            for _, row in pairs[[b_col, a_col]].iterrows():
                ax.plot([0, 1], row.values, color="#6c757d", alpha=0.35)
                ax.scatter([0, 1], row.values, color="#d62728", s=26, zorder=3)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([b_col, a_col])
            ax.set_title(title)
            ax.grid(alpha=0.2)
            _annotate_stat(ax, metric, stage)
        fig.tight_layout()
        paired_path = base.with_name(base.stem + f"{suffix}_paired.png")
        fig.savefig(paired_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(paired_path)

        melt_df = stage_summary.melt(id_vars=["subject", "condition", "stage"], value_vars=["rmssd", "lf", "hf", "lf_hf"], var_name="metric", value_name="value")
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
        metrics = ["rmssd", "hf", "lf", "lf_hf"]
        for ax, metric in zip(axes2.ravel(), metrics):
            sub = melt_df[melt_df["metric"] == metric].dropna()
            if sub.empty:
                ax.set_title(f"{metric}: pas de données")
                ax.axis("off")
                continue
            order = _condition_order(sub["condition"])
            sns.violinplot(data=sub, x="condition", y="value", inner="box", ax=ax, palette="Set2", order=order)
            title = metric.upper() if metric != "lf_hf" else "LF/HF (interprétatif)"
            if multi_stage:
                title = f"{title} – {stage}"
            ax.set_title(title)
            ax.grid(alpha=0.2)
            _annotate_stat(ax, metric, stage)
        fig2.tight_layout()
        dist_path = base.with_name(base.stem + f"{suffix}_dist.png")
        fig2.savefig(dist_path, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        out_paths.append(dist_path)

        fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
        for ax, metric in zip(axes3.ravel(), metrics):
            sub = stage_summary[["condition", metric]].dropna()
            if sub.empty:
                ax.set_title(f"{metric}: pas de données")
                ax.axis("off")
                continue
            order = _condition_order(sub["condition"])
            sns.histplot(data=sub, x=metric, hue="condition", hue_order=order, element="step", stat="density", common_norm=False, ax=ax)
            texts = []
            for cond in order:
                vals = sub.loc[sub["condition"] == cond, metric].dropna()
                if not vals.empty:
                    med = float(vals.median())
                    mad = float(np.median(np.abs(vals - med)))
                    texts.append(f"{cond}: med={med:.3f}, MAD={mad:.3f}")
            if texts:
                ax.text(0.02, 0.98, "\n".join(texts), ha="left", va="top", transform=ax.transAxes, fontsize=8, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
            title = f"Histogramme {metric.upper()}" if metric != "lf_hf" else "Histogramme LF/HF (interprétatif)"
            if multi_stage:
                title = f"{title} – {stage}"
            ax.set_title(title)
            ax.grid(alpha=0.2)
        fig3.tight_layout()
        hist_path = base.with_name(base.stem + f"{suffix}_hist.png")
        fig3.savefig(hist_path, dpi=200, bbox_inches="tight")
        plt.close(fig3)
        out_paths.append(hist_path)

        if not stage_epochs.empty and "duration_s" in stage_epochs.columns:
            fig4, ax4 = plt.subplots(figsize=(10, 4))
            order_seg = _condition_order(stage_epochs["condition"])
            sns.boxplot(data=stage_epochs, x="condition", y="duration_s", ax=ax4, palette="Pastel1", order=order_seg)
            ax4.set_title(f"Durée des segments HRV (s){' – ' + stage if multi_stage else ''}")
            ax4.grid(alpha=0.2)
            dur_path = base.with_name(base.stem + f"{suffix}_segments.png")
            fig4.tight_layout()
            fig4.savefig(dur_path, dpi=200, bbox_inches="tight")
            plt.close(fig4)
            out_paths.append(dur_path)

        if not stage_epochs.empty and "rr_reject_pct" in stage_epochs.columns:
            fig5, axes5 = plt.subplots(1, 2, figsize=(12, 4))
            sub_rr = stage_epochs[["condition", "rr_median"]].dropna()
            if not sub_rr.empty:
                order_rr = _condition_order(sub_rr["condition"])
                sns.histplot(data=sub_rr, x="rr_median", hue="condition", hue_order=order_rr, element="step", stat="density", common_norm=False, ax=axes5[0])
                axes5[0].set_title(f"Distribution médiane RR (s){' – ' + stage if multi_stage else ''}")
                axes5[0].grid(alpha=0.2)
            else:
                axes5[0].set_title("RR médian: pas de données")
                axes5[0].axis("off")
            sub_rej = stage_epochs[["condition", "rr_reject_pct"]].dropna()
            if not sub_rej.empty:
                order_rej = _condition_order(sub_rej["condition"])
                sns.boxplot(data=sub_rej, x="condition", y="rr_reject_pct", ax=axes5[1], palette="Pastel2", order=order_rej)
                axes5[1].set_title(f"% RR rejetés{' – ' + stage if multi_stage else ''}")
                axes5[1].grid(alpha=0.2)
            else:
                axes5[1].set_title("% RR rejetés: pas de données")
                axes5[1].axis("off")
            fig5.tight_layout()
            rr_path = base.with_name(base.stem + f"{suffix}_rr_quality.png")
            fig5.savefig(rr_path, dpi=200, bbox_inches="tight")
            plt.close(fig5)
            out_paths.append(rr_path)

    LOGGER.info("[HRV] Figures sauvegardées: %s", ", ".join(str(p) for p in out_paths))
    return out_paths, stats_df


def plot_stage_profiles(
    result: GroupAnalysisResult,
    stage: str,
    *,
    before_label: Optional[str] = None,
    after_label: Optional[str] = None,
    ax=None,
) -> object:
    """Trace un spaghetti plot stage donné (retourne l'axe Matplotlib)."""

    import matplotlib.pyplot as plt  # import local pour réduire temps de chargement

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    stage_norm = normalise_stage(stage)
    df = result.profiles[result.profiles["stage"] == stage_norm]
    if df.empty:
        ax.set_title(f"Aucune donnée pour {stage_norm}")
        return ax

    before = (before_label or result.config.before_label or "").strip() or "AVANT"
    after = (after_label or result.config.after_label or "").strip() or "APRÈS"

    available_conditions = [str(cond) for cond in df["condition"].dropna().unique()]
    if available_conditions:
        if before not in available_conditions:
            before = available_conditions[0]
        if after not in available_conditions or after == before:
            fallback = next((c for c in available_conditions if c != before), available_conditions[0])
            after = fallback

    # EXCLURE tau=0 (CI) des graphiques - ce n'est pas une échelle d'entropie
    df_filtered = df[df["tau"] > 0].copy()
    taus = sorted(df_filtered["tau"].unique())
    colors = plt.get_cmap("tab20")
    for idx, subject in enumerate(sorted(df_filtered["subject"].unique())):
        subj_df = df_filtered[df_filtered["subject"] == subject]
        curve_b = subj_df[subj_df["condition"] == before].sort_values("tau")
        curve_a = subj_df[subj_df["condition"] == after].sort_values("tau")
        if not curve_b.empty:
            ax.plot(
                curve_b["tau"],
                curve_b["value"],
                color=colors(idx % 20),
                alpha=0.35,
                linestyle="--",
                linewidth=1.0,
            )
        if not curve_a.empty:
            ax.plot(
                curve_a["tau"],
                curve_a["value"],
                color=colors(idx % 20),
                alpha=0.85,
                linewidth=1.5,
                label=subject if idx < 10 else None,
            )

    # Moyennes globales (exclure tau=0)
    mean_b = df_filtered[df_filtered["condition"] == before].groupby("tau")["value"].mean().reindex(taus)
    mean_a = df_filtered[df_filtered["condition"] == after].groupby("tau")["value"].mean().reindex(taus)
    if mean_b.notna().any():
        ax.plot(taus, mean_b, color="#6c757d", linewidth=2.5, linestyle="--", label=f"Moy. {before}")
    if mean_a.notna().any():
        ax.plot(taus, mean_a, color="#dc3545", linewidth=3.0, label=f"Moy. {after}")

    ax.set_xlabel("Échelle τ")
    ax.set_ylabel("Entropie")
    ax.set_title(f"Profil {result.config.metric.upper()} – Stade {stage_norm}")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    return ax


# ============================================================================
# Export helpers
# ============================================================================


def _write_csv_with_metadata(path: str | Path, df: pd.DataFrame, config: GroupAnalysisConfig) -> None:
    metadata_lines = _build_export_metadata(config)
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        for line in metadata_lines:
            handle.write(f"# {line}\n")
        df.to_csv(handle, index=False)


def _build_export_metadata(config: GroupAnalysisConfig) -> List[str]:
    channels = config.display_channels
    if not channels and config.channel_names:
        channels = ", ".join(config.channel_names)
    metadata = [
        f"Metric: {config.metric.upper()}",
        f"Label avant: {config.display_before_label or config.before_label}",
        f"Label après: {config.display_after_label or config.after_label}",
    ]
    if channels:
        metadata.append(f"Canaux: {channels}")
    stages = ", ".join(config.normalised_stage_list())
    if stages:
        metadata.append(f"Stades: {stages}")
    metadata.append(f"Epoch (s): {config.epoch_seconds}")
    if config.metric.lower() == "mse":
        metadata.append(
            f"MSE paramètres: m={config.mse_config.m}, r={config.mse_config.r}, "
            f"τ={min(config.mse_config.scales)}-{max(config.mse_config.scales)}"
        )
    else:
        metadata.append(
            f"REN paramètres: fenêtre={config.renorm_config.window_length}s, "
            f"overlap={config.renorm_config.overlap}, ordre={config.renorm_config.moment_order}"
        )
    return metadata


# ============================================================================
# Helpers internes
# ============================================================================


def _process_entry(
    entry: GroupDesignEntry,
    config: GroupAnalysisConfig,
    stage_list: Sequence[str],
    entry_idx: Optional[int] = None,
    total_entries: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    prefix = f"{entry.subject} / {entry.condition}"
    if entry_idx is not None and total_entries is not None:
        print(
            f"📁 CHECKPOINT GROUP: {prefix} ({entry_idx}/{total_entries}) -> {entry.edf_path}",
            flush=True,
        )

    records: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    if not entry.edf_path.exists():
        LOGGER.warning("EDF introuvable: %s", entry.edf_path)
        return records, summary_rows

    raw = None
    try:
        raw = open_raw_file(str(entry.edf_path), preload=False, verbose="ERROR")  # type: ignore[arg-type]
    except Exception as exc:
        LOGGER.error("Lecture EDF échouée (%s): %s", entry.edf_path, exc)
        return records, summary_rows

    try:
        sfreq = float(raw.info.get("sfreq", 0.0))
        if sfreq <= 0:
            LOGGER.warning("Fréquence invalide pour %s", entry.edf_path)
            return records, summary_rows

        picks = _select_picks(raw, config.channel_names)
        scoring_df = _load_scoring(entry, raw, config.epoch_seconds)

        stages_for_entry = [
            stage
            for stage in entry.resolved_stages(stage_list)
            if stage in stage_list
        ]
        if not stages_for_entry:
            stages_for_entry = stage_list

        if config.metric.lower() == "mse":
            # NOUVELLE LOGIQUE: MSE par epoch puis moyenne par stade (comme script externe)
            epoch_results_by_stage: Dict[str, List[Dict[int, float]]] = {}
            
            if scoring_df is None or scoring_df.empty:
                pass  # Pas de scoring, on ne peut pas faire epoch par epoch
            else:
                samples_per_epoch = int(round(config.epoch_seconds * sfreq))
                
                # Filter epochs for stages we care about
                relevant_rows = [
                    (idx, row) for idx, row in scoring_df.iterrows()
                    if str(row["stage"]) in stages_for_entry or "ALL" in stages_for_entry
                ]
                total_epochs = len(relevant_rows)
                total_duration = float(raw.n_times / sfreq) if raw.n_times > 0 else 0.0
                
                if total_epochs > 0:
                    print(
                        f"📊 CHECKPOINT GROUP: Processing {total_epochs} epoch(s) "
                        f"(total duration: {total_duration:.1f}s)",
                        flush=True,
                    )
                
                epoch_count = 0
                cumulative_time = 0.0
                
                for _, row in relevant_rows:
                    stage = str(row["stage"])
                    if stage not in stages_for_entry and "ALL" not in stages_for_entry:
                        continue
                    
                    t0 = float(row["time"])
                    start = int(max(0, min(raw.n_times - 1, math.floor(t0 * sfreq))))
                    stop = int(min(raw.n_times, start + samples_per_epoch))
                    
                    if stop <= start:
                        continue
                    
                    epoch_data = raw.get_data(picks=picks, start=start, stop=stop)
                    if epoch_data.size == 0:
                        continue
                    
                    epoch_count += 1
                    epoch_duration = (stop - start) / sfreq
                    cumulative_time += epoch_duration
                    
                    # Calculer MSE sur cette epoch unique (aplatir canaux comme script externe)
                    try:
                        label = f"{entry.subject}/{entry.condition} – {stage} epoch@{t0:.1f}s"
                        epoch_mse, epoch_ci = _compute_mse_profile(epoch_data, sfreq, config, label, flatten_channels=True)
                        
                        if stage not in epoch_results_by_stage:
                            epoch_results_by_stage[stage] = []
                        # Stocker à la fois les valeurs par tau ET le CI
                        epoch_results_by_stage[stage].append({"tau_values": epoch_mse, "ci": epoch_ci})
                        
                        # Log epoch processing with progress (only every 50 epochs or at milestones)
                        if epoch_count % 50 == 0 or epoch_count == total_epochs:
                            valid_taus = [tau for tau, val in epoch_mse.items() if not math.isnan(val)]
                            print(
                                f"✅ CHECKPOINT GROUP: Epoch {epoch_count}/{total_epochs} "
                                f"({cumulative_time:.1f}s/{total_duration:.1f}s) - "
                                f"stage {stage} @ {t0:.1f}s - {len(valid_taus)} tau values, CI={epoch_ci:.4f}",
                                flush=True,
                            )
                    except Exception as exc:
                        LOGGER.warning(
                            "Erreur MSE epoch %s @ %.1fs (%d/%d, %.1fs/%.1fs): %s",
                            stage, t0, epoch_count, total_epochs, cumulative_time, total_duration, exc
                        )
                        continue
            
            # Moyenner les résultats par stade
            for stage, epoch_results_list in epoch_results_by_stage.items():
                if not epoch_results_list:
                    LOGGER.debug("No epochs processed for stage %s", stage)
                    continue
                
                # Collecter tous les tau présents
                all_taus = set()
                for epoch_result in epoch_results_list:
                    all_taus.update(epoch_result["tau_values"].keys())
                
                if not all_taus:
                    LOGGER.debug("No tau values found for stage %s", stage)
                    continue
                
                sorted_tau = sorted(all_taus)
                # Moyenne par tau sur toutes les epochs
                averaged_values = {}
                for tau in sorted_tau:
                    tau_values = [
                        epoch_result["tau_values"].get(tau, math.nan)
                        for epoch_result in epoch_results_list
                        if tau in epoch_result["tau_values"]
                    ]
                    if tau_values:
                        valid_tau_values = [v for v in tau_values if not math.isnan(v)]
                        if valid_tau_values:
                            averaged_values[tau] = float(np.nanmean(valid_tau_values))
                
                # Moyenne du CI sur toutes les epochs
                ci_values = [
                    epoch_result["ci"]
                    for epoch_result in epoch_results_list
                    if not math.isnan(epoch_result["ci"])
                ]
                averaged_ci = float(np.nanmean(ci_values)) if ci_values else math.nan
                
                # Log aggregation summary (reduced verbosity - only at completion)
                if len(epoch_results_list) > 0:
                    LOGGER.debug(
                        "MSE aggregation for %s/%s - stage %s: %d epochs, "
                        "%d tau values averaged, CI=%.4f",
                        entry.subject, entry.condition, stage,
                        len(epoch_results_list), len(averaged_values), averaged_ci
                    )
                
                # Enregistrer les résultats moyens par tau (EXCLURE tau=0 qui n'est pas une vraie échelle)
                # Le CI est stocké séparément dans le summary, pas dans les records de profils
                for tau, value in averaged_values.items():
                    if tau > 0:  # Exclure tau=0 (qui ne devrait pas être dans averaged_values de toute façon)
                        records.append(
                            {
                                "subject": entry.subject,
                                "condition": entry.condition,
                                "stage": stage,
                                "tau": float(tau),
                                "value": float(value),
                            }
                        )
                
                # IMPORTANT: Ne PAS stocker le CI avec tau=0 dans les records
                # Le CI est une métrique différente (somme des entropies) et ne doit pas être mélangé
                # avec les valeurs d'entropie par échelle. Il est stocké uniquement dans le summary.
                
                # Calculer AUC et moyenne en excluant explicitement tau=0 si présent
                valid_taus_for_auc = [tau for tau in sorted_tau if tau > 0]
                values = [averaged_values[tau] for tau in valid_taus_for_auc if tau in averaged_values]
                auc = float(np.trapz(values, valid_taus_for_auc)) if len(valid_taus_for_auc) > 1 else (float(values[0]) if values else float(np.nan))
                summary_rows.append(
                    {
                        "subject": entry.subject,
                        "condition": entry.condition,
                        "stage": stage,
                        "auc": auc,
                        "mean": float(np.mean(values)),
                        "ci": float(averaged_ci),  # Ajouter CI au summary
                    }
                )
        else:
            # LOGIQUE ORIGINALE pour renorm (concaténation par stade)
            for stage in stages_for_entry:
                stage_data = _extract_stage_data(
                    raw, scoring_df, stage, config.epoch_seconds, picks, config.channel_names
                )

                if stage_data is None or stage_data.size == 0:
                    continue

                try:
                    metric_map = _compute_renorm_profile(stage_data, sfreq, config)
                except Exception as exc:
                    LOGGER.error("Erreur calcul %s – %s (%s): %s", entry.subject, stage, config.metric, exc)
                    continue

                sorted_tau = sorted(metric_map)
                if not sorted_tau:
                    continue

                values = [metric_map[tau] for tau in sorted_tau]
                for tau, value in zip(sorted_tau, values):
                    records.append(
                        {
                            "subject": entry.subject,
                            "condition": entry.condition,
                            "stage": stage,
                            "tau": float(tau),
                            "value": float(value),
                        }
                    )

                auc = float(np.trapz(values, sorted_tau)) if len(sorted_tau) > 1 else float(values[0])
                summary_rows.append(
                    {
                        "subject": entry.subject,
                        "condition": entry.condition,
                        "stage": stage,
                        "auc": auc,
                        "mean": float(np.mean(values)),
                    }
                )
    finally:
        if raw is not None:
            raw.close()

    if entry_idx is not None and total_entries is not None:
        print(
            f"✅ CHECKPOINT GROUP: Terminé {prefix} ({entry_idx}/{total_entries})",
            flush=True,
        )

    return records, summary_rows


def _notify(cb: Optional[Callable[[str, float], None]], message: str, progress: float) -> None:
    if cb is None:
        return
    try:
        cb(message, max(0.0, min(1.0, progress)))
    except Exception:
        pass


def normalise_stage(stage: str) -> str:
    return _STAGE_ALIASES.get(str(stage).strip().upper(), str(stage).strip().upper())


def normalise_condition_label(label: str) -> str:
    """Normalise un label de condition pour comparer AV/Après ↔ Pre/Post."""
    decomposed = unicodedata.normalize("NFKD", str(label or ""))
    ascii_only = decomposed.encode("ASCII", "ignore").decode("ascii")
    compact = re.sub(r"[^a-z0-9]+", "", ascii_only.lower())
    return compact


def _match_condition_column(columns: Iterable[str], target: str) -> Optional[str]:
    """Retourne le nom exact de colonne correspondant au label cible, en tenant compte des accents et espaces."""
    if not target:
        return None
    wanted = normalise_condition_label(target)
    if not wanted:
        return None
    for col in columns:
        if normalise_condition_label(col) == wanted:
            return col
    return None


def _select_picks(raw, channel_names: Optional[Sequence[str]]) -> Optional[Sequence[int]]:
    if not channel_names:
        return None
    picks = []
    for name in channel_names:
        if name in raw.ch_names:
            picks.append(raw.ch_names.index(name))
    if not picks:
        LOGGER.warning("Aucun des canaux demandés (%s) n'est présent dans %s", channel_names, raw.filenames)
        return None
    return picks


def _load_scoring(entry: GroupDesignEntry, raw, epoch_seconds: float) -> Optional[pd.DataFrame]:
    if entry.scoring_path:
        scoring_path = entry.scoring_path
        if not scoring_path.exists():
            LOGGER.warning("Scoring introuvable: %s", scoring_path)
            return None
        suffix = scoring_path.suffix.lower()
        # Point de référence absolu pour aligner le scoring sur l'EDF
        meas_date = raw.info.get("meas_date")
        try:
            base_ts = pd.Timestamp(meas_date) if meas_date is not None else None
        except Exception:  # pragma: no cover - robuste aux formats exotiques
            base_ts = None
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(scoring_path)
            scoring = import_excel_scoring(df, absolute_start_datetime=base_ts, epoch_seconds=epoch_seconds)
        elif suffix in {".csv"}:
            df = pd.read_csv(scoring_path)
            scoring = import_excel_scoring(df, absolute_start_datetime=base_ts, epoch_seconds=epoch_seconds)
        elif suffix in {".edf", ".edf+"}:
            duration = float(raw.n_times / raw.info["sfreq"])
            scoring = import_edf_hypnogram(
                str(scoring_path),
                recording_duration_s=duration,
                epoch_seconds=epoch_seconds,
                absolute_start_datetime=base_ts,
            )
        else:
            LOGGER.warning("Format scoring non supporté (%s).", scoring_path)
            scoring = None
    else:
        scoring = None

    if scoring is None or scoring.empty:
        return None

    scoring = scoring.copy()
    scoring["stage"] = scoring["stage"].astype(str).map(normalise_stage)
    return scoring


def _extract_stage_data(
    raw,
    scoring_df: Optional[pd.DataFrame],
    stage: str,
    epoch_seconds: float,
    picks: Optional[Sequence[int]],
    channel_names: Optional[Sequence[str]],
) -> Optional[np.ndarray]:
    """Retourne les données concaténées pour un stade donné."""

    fs = float(raw.info["sfreq"])
    if stage == "ALL":
        data = raw.get_data(picks=picks)
        return data.astype(np.float32, copy=False)

    if scoring_df is None or scoring_df.empty:
        return None

    stage_mask = scoring_df["stage"] == stage
    if not stage_mask.any():
        return None

    samples_per_epoch = int(round(epoch_seconds * fs))
    segments: List[np.ndarray] = []
    for t0 in scoring_df.loc[stage_mask, "time"].astype(float):
        start = int(max(0, min(raw.n_times - 1, math.floor(t0 * fs))))
        stop = int(min(raw.n_times, start + samples_per_epoch))
        if stop <= start:
            continue
        seg = raw.get_data(picks=picks, start=start, stop=stop)
        segments.append(seg)

    if not segments:
        return None
    concat = np.concatenate(segments, axis=1)
    return concat.astype(np.float32, copy=False)


def _compute_mse_profile(
    data: np.ndarray,
    sfreq: float,
    config: GroupAnalysisConfig,
    progress_label: Optional[str] = None,
    *,
    flatten_channels: bool = False,
) -> Tuple[Dict[int, float], float]:
    """Compute MSE profile. If flatten_channels=True, flatten all channels into 1D signal (like external script).
    
    Returns:
        Tuple of (entropy_by_scale dict, complexity_index float)
    """
    prefix = f"[{progress_label}] " if progress_label else ""
    
    channel_names_for_mse = None
    if flatten_channels and data.ndim == 2 and data.shape[0] > 1:
        # Aplatir tous les canaux en un seul signal 1D (comme script externe)
        original_n_channels = data.shape[0]
        data = data.flatten()
        data = data[np.newaxis, :]  # Remettre en 2D (1 channel, n_samples)
        # Nommer le canal aplati de manière descriptive
        channel_names_for_mse = (f"flattened_{original_n_channels}ch",)
    elif data.ndim == 2:
        # Si on a plusieurs canaux mais qu'on ne les aplatit pas, utiliser des noms génériques
        channel_names_for_mse = tuple(f"ch{idx}" for idx in range(data.shape[0]))
    
    res = compute_multiscale_entropy(
        data,
        sfreq=sfreq,
        config=config.mse_config,
        progress_label=progress_label,
        channel_names=channel_names_for_mse,
    )
    
    # Extraire le CI depuis les détails (moyenne des CI des canaux)
    ci_values = []
    for scale_detail in res.sample_entropy_by_scale.values():
        if isinstance(scale_detail, dict) and "channel_details" in scale_detail:
            for ch_detail in scale_detail["channel_details"].values():
                if isinstance(ch_detail, dict) and "complexity_index" in ch_detail:
                    ci_values.append(float(ch_detail["complexity_index"]))
    
    complexity_index = float(np.nanmean(ci_values)) if ci_values else math.nan
    
    # Reduced verbosity: removed per-epoch profile log
    
    return dict(res.entropy_by_scale), complexity_index


def _compute_renorm_profile(data: np.ndarray, sfreq: float, config: GroupAnalysisConfig) -> Dict[int, float]:
    channel_names = tuple(f"ch{idx}" for idx in range(data.shape[0]))
    res = compute_renormalized_entropy(data, sfreq=sfreq, channel_names=channel_names, config=config.renorm_config)
    # Utiliser l'entropie principale comme unique valeur (tau=0)
    return {0: float(res.entropy_nats)}


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Applique la correction de Benjamini-Hochberg."""

    p = np.asarray(p_values, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = np.empty_like(p)
    prev = 0.0
    for i, idx in enumerate(order, start=1):
        value = p[idx] * n / i
        value = min(value, 1.0)
        prev = max(prev, value)
        ranked[idx] = prev
    return np.minimum(ranked, 1.0)


__all__ = [
    "GroupDesignEntry",
    "GroupAnalysisConfig",
    "GroupAnalysisResult",
    "StatsConfig",
    "GroupAnalysisError",
    "MultiscaleEntropyConfig",
    "RenormalizedEntropyConfig",
    "load_design_csv",
    "parse_stage_field",
    "compute_group_profiles",
    "run_statistical_tests",
    "export_profiles_to_csv",
    "export_stats_to_csv",
    "plot_stage_profiles",
    "benjamini_hochberg",
]
