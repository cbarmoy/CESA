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
from .scoring_io import import_excel_scoring, import_edf_hypnogram
from .advanced_spaghetti_plots import (
    permutation_test_median_diff,
    bootstrap_ci_median_diff,
    robust_z_intrasubject,
)

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
    max_workers: int = 0  # 0 => auto (calqué sur le nombre de cœurs disponibles)

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
        taus = sorted(self.profiles["tau"].unique().tolist())
        return taus


class GroupAnalysisError(RuntimeError):
    """Exception spécifique à l'analyse de groupe."""


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
    desired_workers = config.max_workers if config.max_workers and config.max_workers > 0 else cpu_count
    workers = max(1, min(desired_workers, total_entries))
    print(
        f"⚙️ CHECKPOINT GROUP: utilisation de {workers} worker(s) (CPU dispo: {cpu_count})",
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
            f"[STATS] Label avant saisi « {cfg.display_before_label} » → colonne réelle « {cfg.before_label} ».",
            flush=True,
        )
    if cfg.display_after_label and normalise_condition_label(cfg.display_after_label) != normalise_condition_label(cfg.after_label):
        print(
            f"[STATS] Label après saisi « {cfg.display_after_label} » → colonne réelle « {cfg.after_label} ».",
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
        for tau in sorted(stage_df["tau"].unique()):
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

    taus = sorted(df["tau"].unique())
    colors = plt.get_cmap("tab20")
    for idx, subject in enumerate(sorted(df["subject"].unique())):
        subj_df = df[df["subject"] == subject]
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

    # Moyennes globales
    mean_b = df[df["condition"] == before].groupby("tau")["value"].mean().reindex(taus)
    mean_a = df[df["condition"] == after].groupby("tau")["value"].mean().reindex(taus)
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
        raw = mne.io.read_raw_edf(str(entry.edf_path), preload=False, verbose="ERROR")  # type: ignore[arg-type]
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

        for stage in stages_for_entry:
            stage_data = _extract_stage_data(
                raw, scoring_df, stage, config.epoch_seconds, picks, config.channel_names
            )

            if stage_data is None or stage_data.size == 0:
                continue

            try:
                if config.metric.lower() == "renorm":
                    metric_map = _compute_renorm_profile(stage_data, sfreq, config)
                else:
                    label = f"{entry.subject}/{entry.condition} – {stage}"
                    metric_map = _compute_mse_profile(stage_data, sfreq, config, label)
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
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(scoring_path)
            scoring = import_excel_scoring(df, absolute_start_datetime=None, epoch_seconds=epoch_seconds)
        elif suffix in {".csv"}:
            df = pd.read_csv(scoring_path)
            scoring = import_excel_scoring(df, absolute_start_datetime=None, epoch_seconds=epoch_seconds)
        elif suffix in {".edf", ".edf+"}:
            duration = float(raw.n_times / raw.info["sfreq"])
            scoring = import_edf_hypnogram(str(scoring_path), recording_duration_s=duration, epoch_seconds=epoch_seconds)
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
) -> Dict[int, float]:
    res = compute_multiscale_entropy(
        data,
        sfreq=sfreq,
        config=config.mse_config,
        progress_label=progress_label,
    )
    return dict(res.entropy_by_scale)


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
    "load_design_csv",
    "parse_stage_field",
    "compute_group_profiles",
    "run_statistical_tests",
    "export_profiles_to_csv",
    "export_stats_to_csv",
    "plot_stage_profiles",
    "benjamini_hochberg",
]

