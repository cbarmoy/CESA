#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génération de graphiques spaghetti intra-sujet directement à partir des FFT (CSV).

Fonctionnalités:
- Canaux: tous les canaux EEG disponibles (filtrables)
- Bandes: Delta, Theta, Alpha, Beta, Gamma (filtrables)
- Stades: W, N1, N2, N3, R (filtrables)
- Méthodes robustes intra-sujet (même statistique de base: median(diff)):
  • Permutation sign-flip appariée (fallback non apparié)
  • Bootstrap CI sur median(diff)
  • Z robuste basé sur MAD des différences
- Décision par méthode: augmentation / diminution / stagnation
- Consensus ≥ 2/3: annotation "★" sur le graphique
- Tableau latéral intégré (médianes avant/après, D_obs, p, CI, Z, décisions, consensus)
- Export CSV des données utilisées pour chaque graphique

Entrée recommandée (par ligne):
  Subject, Channel, Group (AVANT/APRÈS), stage, Frequency, Amplitude, [Epoch?]

Deux APIs principales:
- generate_spaghetti_from_fft_csv(before_dir, after_dir, output_dir, ...)
- generate_spaghetti_from_dataframe(df, output_dir, ...)
"""

from __future__ import annotations

import os
import glob
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import logging
import time
 
try:
    import mne  # type: ignore
except Exception:
    mne = None  # mne est requis pour lecture EDF

try:
    from . import scoring_io  # type: ignore
except Exception:
    scoring_io = None

# Réutiliser les bandes standards du module spectral_analysis si disponible
try:
    from .spectral_analysis import EEG_BANDS as DEFAULT_BANDS  # type: ignore
except Exception:
    DEFAULT_BANDS = {
        "LowDelta": (0.3, 1.0),
        "Delta": (1.0, 4.0),
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 12.0),
        "Sigma": (12.0, 15.0),
        "Beta": (15.0, 30.0),
        "Gamma": (30.0, 45.0),
    }


# =====================
# Paramètres globaux
# =====================
RNG_SEED = 42
N_PERM = 5000
N_BOOT = 2000
ALPHA = 0.05
ROBUST_Z_THRESHOLD = 2.5


# =====================
# Utilitaires d'E/S
# =====================
def _extract_channel_from_filename(filename: str) -> str:
    name_no_ext = Path(filename).stem
    m = re.search(r"[A-Z]\d_[A-Z]\d", name_no_ext)
    if m:
        return m.group()
    m = re.search(r"[A-Z]\d[A-Z]\d", name_no_ext)
    if m:
        return m.group()
    parts = name_no_ext.split("_")
    for p in parts:
        if re.match(r"^[A-Z]\d_[A-Z]\d$", p):
            return p
        if re.match(r"^[A-Z]\d[A-Z]\d$", p):
            return p
    return name_no_ext


def _load_fft_group(folder_path: str, group_name: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {folder_path}")
    rows: List[pd.DataFrame] = []
    for fp in csv_files:
        try:
            df = pd.read_csv(fp)
            if len(df) == 0:
                continue
            df = df.rename(columns={
                'freq_hz': 'Frequency',
                'magnitude': 'Amplitude',
            })
            if 'Frequency' not in df.columns or 'Amplitude' not in df.columns:
                continue
            filename = os.path.basename(fp)
            df['Subject'] = Path(filename).stem
            df['Channel'] = _extract_channel_from_filename(filename)
            df['Group'] = group_name
            df['Filename'] = filename
            rows.append(df)
        except Exception:
            continue
    if not rows:
        raise ValueError(f"Aucune donnée valide dans {folder_path}")
    return pd.concat(rows, ignore_index=True)


# =====================
# Calculs de bande et métriques
# =====================
def _integrate_auc_sorted(freq: np.ndarray, amp: np.ndarray) -> float:
    return float(np.trapz(amp, freq))


def _assign_band(freq_value: float, bands: Dict[str, Tuple[float, float]]) -> Optional[str]:
    for name, (lo, hi) in bands.items():
        if (freq_value >= lo) and (freq_value <= hi):
            return name
    return None


def compute_band_metrics_from_fft(
    data: pd.DataFrame,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    total_range: Tuple[float, float] = (0.5, 100.0),
) -> pd.DataFrame:
    """
    Calcule par Subject/Channel/Group/Stage[/Epoch] la puissance relative de bande
    à partir des FFT (AUC bande / AUC totale). Retourne un DataFrame avec colonnes:
      Subject, Channel, Group, stage (si dispo), Epoch (si dispo), Band, AUC, Peak_Amplitude, Peak_Frequency, Barycenter
    """
    if bands is None:
        bands = DEFAULT_BANDS
    df = data.copy()
    # Colonnes standard
    df = df.rename(columns={'Stage': 'stage'})
    if 'stage' not in df.columns:
        df['stage'] = 'ALL'
    else:
        # Canonicaliser les stades vers W, N1, N2, N3, R
        mapping = {
            'W': 'W', 'WAKE': 'W', 'AWAKE': 'W', 'EVEIL': 'W', 'ÉVEIL': 'W',
            'R': 'R', 'REM': 'R', 'PARADOXAL': 'R', 'SOMMEIL PARADOXAL': 'R',
            'N1': 'N1', 'S1': 'N1', 'NREM1': 'N1', 'STAGE1': 'N1', 'STAGE 1': 'N1',
            'N2': 'N2', 'S2': 'N2', 'NREM2': 'N2', 'STAGE2': 'N2', 'STAGE 2': 'N2',
            'N3': 'N3', 'S3': 'N3', 'S4': 'N3', 'NREM3': 'N3', 'NREM4': 'N3', 'STAGE3': 'N3', 'STAGE 3': 'N3', 'STAGE4': 'N3', 'STAGE 4': 'N3',
        }
        df['stage'] = df['stage'].astype(str).str.upper().map(mapping).fillna('ALL')
    epoch_col = None
    for cand in ['epoch', 'Epoch', 'epoques', 'Epoque', 'window', 'Window', 'segment', 'Segment', 'trial', 'Trial']:
        if cand in df.columns:
            epoch_col = cand
            break
    if epoch_col is not None and epoch_col != 'Epoch':
        df = df.rename(columns={epoch_col: 'Epoch'})

    # AUC totale par bloc (Subject, Channel, Group, stage, Epoch?)
    keys_total = ['Subject', 'Channel', 'Group', 'stage'] + (['Epoch'] if 'Epoch' in df.columns else [])
    def _auc_block(block: pd.DataFrame) -> float:
        x = block.sort_values('Frequency')
        x = x[(x['Frequency'] >= total_range[0]) & (x['Frequency'] <= total_range[1])]
        return _integrate_auc_sorted(x['Frequency'].to_numpy(), x['Amplitude'].to_numpy())
    auc_total = df.groupby(keys_total, as_index=False).apply(_auc_block).rename(columns={None: 'AUC_Total'})

    # Assigner bandes et calculs par bande
    df['Band'] = df['Frequency'].apply(lambda f: _assign_band(float(f), bands))
    df = df.dropna(subset=['Band'])
    keys_band = ['Subject', 'Channel', 'Group', 'stage', 'Band'] + (['Epoch'] if 'Epoch' in df.columns else [])

    def _band_metrics(block: pd.DataFrame) -> pd.Series:
        x = block.sort_values('Frequency')
        auc_b = _integrate_auc_sorted(x['Frequency'].to_numpy(), x['Amplitude'].to_numpy())
        peak_amp = float(x['Amplitude'].max())
        idx = int(x['Amplitude'].idxmax())
        peak_freq = float(block.loc[idx, 'Frequency']) if len(block) else np.nan
        denom = float(x['Amplitude'].sum())
        bary = float((x['Frequency'] * x['Amplitude']).sum() / denom) if denom > 0 else np.nan
        return pd.Series({
            'AUC_Band': auc_b,
            'Peak_Amplitude': peak_amp,
            'Peak_Frequency': peak_freq,
            'Barycenter': bary,
        })

    band_metrics = df.groupby(keys_band).apply(_band_metrics).reset_index()
    merged = pd.merge(band_metrics, auc_total, on=['Subject', 'Channel', 'Group', 'stage'] + (['Epoch'] if 'Epoch' in df.columns else []), how='left')
    merged['AUC'] = merged['AUC_Band'] / merged['AUC_Total']
    return merged


# =====================
# Méthodes robustes intra-sujet (median(diff))
# =====================
def _median_diff(x_before: np.ndarray, x_after: np.ndarray) -> Tuple[float, float, float]:
    med_b = float(np.median(x_before)) if x_before.size else np.nan
    med_a = float(np.median(x_after)) if x_after.size else np.nan
    return med_a - med_b, med_b, med_a


def permutation_test_median_diff(
    x_before: np.ndarray,
    x_after: np.ndarray,
    n_perm: int = N_PERM,
    random_state: int = RNG_SEED,
) -> Dict[str, float | str]:
    x_before = np.asarray(x_before, dtype=float)
    x_after = np.asarray(x_after, dtype=float)
    d_obs, med_b, med_a = _median_diff(x_before, x_after)
    n1, n2 = x_before.size, x_after.size

    # Cas limites: données insuffisantes OU 1 seule époque -> stagnation
    if n1 == 0 or n2 == 0 or np.isnan(d_obs) or (n1 == 1 and n2 == 1):
        return {
            'D_obs': float(d_obs) if not np.isnan(d_obs) else np.nan,
            'Median_Avant': float(med_b) if not np.isnan(med_b) else np.nan,
            'Median_Apres': float(med_a) if not np.isnan(med_a) else np.nan,
            'p_value': np.nan,
            'decision': 'stagnation',
        }

    rng = np.random.default_rng(random_state)
    if n1 == n2:
        diffs = x_after - x_before
        d0 = float(np.median(diffs))
        count = 0
        for _ in range(n_perm):
            signs = rng.choice([-1, 1], size=n1)
            d_perm = float(np.median(signs * diffs))
            if abs(d_perm) >= abs(d0):
                count += 1
        p_val = count / n_perm
    else:
        d0 = float(d_obs)
        combined = np.concatenate([x_before, x_after])
        count = 0
        for _ in range(n_perm):
            perm = rng.permutation(combined)
            x1 = perm[:n1]
            x2 = perm[n1:]
            d_perm = float(np.median(x2) - np.median(x1))
            if abs(d_perm) >= abs(d0):
                count += 1
        p_val = count / n_perm

    if not np.isnan(d0) and p_val < ALPHA:
        decision = 'augmentation' if d0 > 0 else 'diminution'
    else:
        decision = 'stagnation'

    # Debug
    try:
        print(f"[PERMUTATION] D_obs={d0:.6f}, p={p_val:.4f}, décision={decision}")
    except Exception:
        pass

    return {
        'D_obs': float(d0),
        'Median_Avant': float(med_b),
        'Median_Apres': float(med_a),
        'p_value': float(p_val),
        'decision': decision,
    }


def bootstrap_ci_median_diff(
    x_before: np.ndarray,
    x_after: np.ndarray,
    n_boot: int = N_BOOT,
    ci: float = 0.95,
    random_state: int = RNG_SEED,
) -> Dict[str, float | str]:
    x_before = np.asarray(x_before, dtype=float)
    x_after = np.asarray(x_after, dtype=float)
    d_obs, med_b, med_a = _median_diff(x_before, x_after)
    n1, n2 = x_before.size, x_after.size

    if n1 == 0 or n2 == 0 or np.isnan(d_obs) or (n1 == 1 and n2 == 1):
        return {
            'D_obs': float(d_obs) if not np.isnan(d_obs) else np.nan,
            'Median_Avant': float(med_b) if not np.isnan(med_b) else np.nan,
            'Median_Apres': float(med_a) if not np.isnan(med_a) else np.nan,
            'CI_low': np.nan,
            'CI_high': np.nan,
            'decision': 'stagnation',
        }

    rng = np.random.default_rng(random_state)
    stats = np.empty(n_boot, dtype=float)

    if n1 == n2:
        diffs = x_after - x_before
        for i in range(n_boot):
            idx = rng.integers(0, n1, size=n1)
            stats[i] = float(np.median(diffs[idx]))
    else:
        for i in range(n_boot):
            b1 = x_before[rng.integers(0, n1, size=n1)]
            b2 = x_after[rng.integers(0, n2, size=n2)]
            stats[i] = float(np.median(b2) - np.median(b1))

    alpha = 1 - ci
    ci_low = float(np.percentile(stats, 100 * alpha / 2))
    ci_high = float(np.percentile(stats, 100 * (1 - alpha / 2)))

    if ci_low > 0:
        decision = 'augmentation'
    elif ci_high < 0:
        decision = 'diminution'
    else:
        decision = 'stagnation'

    # Debug
    try:
        print(f"[BOOTSTRAP] D_obs={d_obs:.6f}, CI=[{ci_low:.6f}, {ci_high:.6f}], décision={decision}")
    except Exception:
        pass

    return {
        'D_obs': float(d_obs),
        'Median_Avant': float(med_b),
        'Median_Apres': float(med_a),
        'CI_low': ci_low,
        'CI_high': ci_high,
        'decision': decision,
    }


def robust_z_intrasubject(
    x_before: np.ndarray,
    x_after: np.ndarray,
    threshold: float = ROBUST_Z_THRESHOLD,
) -> Dict[str, float | str]:
    x_before = np.asarray(x_before, dtype=float)
    x_after = np.asarray(x_after, dtype=float)
    d_obs, med_b, med_a = _median_diff(x_before, x_after)
    n1, n2 = x_before.size, x_after.size

    if n1 == 0 or n2 == 0 or np.isnan(d_obs) or (n1 == 1 and n2 == 1):
        return {
            'D_obs': float(d_obs) if not np.isnan(d_obs) else np.nan,
            'Median_Avant': float(med_b) if not np.isnan(med_b) else np.nan,
            'Median_Apres': float(med_a) if not np.isnan(med_a) else np.nan,
            'Z': np.nan,
            'decision': 'stagnation',
        }

    if n1 == n2:
        diffs = x_after - x_before
        d0 = float(np.median(diffs))
        mad = float(np.median(np.abs(diffs - d0)))
    else:
        rng = np.random.default_rng(RNG_SEED)
        n_boot = 1000
        stats = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx_b = rng.integers(0, n1, size=n1)
            idx_a = rng.integers(0, n2, size=n2)
            stats[i] = float(np.median(x_after[idx_a]) - np.median(x_before[idx_b]))
        d0 = float(np.median(stats))
        mad = float(np.median(np.abs(stats - d0)))

    scale = 1.4826 * mad if mad > 1e-10 else 1e-10
    z = float(d0 / scale)
    if abs(z) > threshold:
        decision = 'augmentation' if z > 0 else 'diminution'
    else:
        decision = 'stagnation'
    # Debug
    try:
        print(f"[ROBUST-Z] D_obs={d0:.6f}, Z={z:.3f}, décision={decision}")
    except Exception:
        pass
    return {
        'D_obs': float(d0),
        'Median_Avant': float(med_b),
        'Median_Apres': float(med_a),
        'Z': float(z),
        'decision': decision,
    }


# =====================
# Tracé spaghetti + tableau latéral
# =====================
def _format_num(x: float) -> str:
    try:
        return f"{float(x):.3f}"
    except Exception:
        return ""


def _build_table_rows(plot_data: pd.DataFrame, metric: str, band: str, stats_df: Optional[pd.DataFrame]) -> Tuple[List[str], List[List[str]], List[Dict[str, str]]]:
    headers = [
        'Sujet', 'Bande', 'Canal', 'Med. avant', 'Med. après', 'D_obs',
        'p (Perm.)', 'CI (Boot.)', 'Z',
        'Perm', 'Boot', 'Zsc', 'Cons.'
    ]
    subjects = plot_data['Subject_Base'].unique().tolist()
    channel = plot_data['Channel'].iloc[0] if 'Channel' in plot_data.columns else ''
    table_rows: List[List[str]] = []
    style_meta: List[Dict[str, str]] = []
    for subj in subjects:
        if stats_df is not None:
            # Utiliser toutes les valeurs par epoch disponibles du même band/canal/stage
            filt = (stats_df['Subject_Base'] == subj) & (stats_df['Band'] == band)
            # Stage/Channel alignés au plot courant
            if 'Channel' in stats_df.columns:
                filt = filt & (stats_df['Channel'] == channel)
            if 'stage' in stats_df.columns and 'stage' in plot_data.columns:
                st_val = plot_data['stage'].iloc[0]
                filt = filt & (stats_df['stage'] == st_val)
            sub_df = stats_df[filt]
        else:
            sub_df = plot_data[plot_data['Subject_Base'] == subj]
        x_b = sub_df[sub_df['Group'] == 'AVANT'][metric].dropna().to_numpy()
        x_a = sub_df[sub_df['Group'] == 'APRÈS'][metric].dropna().to_numpy()
        perm = permutation_test_median_diff(x_b, x_a)
        boot = bootstrap_ci_median_diff(x_b, x_a)
        rz = robust_z_intrasubject(x_b, x_a)
        decs = [perm['decision'], boot['decision'], rz['decision']]
        num_up = sum(d == 'augmentation' for d in decs)
        num_down = sum(d == 'diminution' for d in decs)
        if num_up >= 2:
            cons_dec = 'augmentation'
            cons_text = '★↑'
        elif num_down >= 2:
            cons_dec = 'diminution'
            cons_text = '★↓'
        else:
            cons_dec = 'stagnation' if all(d == 'stagnation' for d in decs) else 'mixte'
            cons_text = '=' if cons_dec == 'stagnation' else '◦'
        
        def _sym(decision: str) -> str:
            d = str(decision)
            if d == 'augmentation':
                return '↑'
            if d == 'diminution':
                return '↓'
            if d == 'mixte':
                return '◦'
            return '='
        row = [
            subj,
            band,
            channel,
            _format_num(perm['Median_Avant']),
            _format_num(perm['Median_Apres']),
            _format_num(perm['D_obs']),
            _format_num(perm.get('p_value', np.nan)),
            f"[{_format_num(boot.get('CI_low', np.nan))}, {_format_num(boot.get('CI_high', np.nan))}]",
            _format_num(rz.get('Z', np.nan)),
            _sym(perm['decision']),
            _sym(boot['decision']),
            _sym(rz['decision']),
            cons_text,
        ]
        table_rows.append(row)
        style_meta.append({'perm': str(perm['decision']), 'boot': str(boot['decision']), 'z': str(rz['decision']), 'cons': cons_dec})
    return headers, table_rows, style_meta


def _color_for_decision(decision: str) -> str:
    if decision == 'augmentation':
        return '#E8F5E9'
    if decision == 'diminution':
        return '#FFEBEE'
    if decision == 'mixte':
        return '#F0F0F0'
    return '#F5F5F5'


def create_spaghetti_plot(
    data: pd.DataFrame,
    metric: str,
    band: str,
    output_dir: str,
    title_suffix: str = "",
    export_csv: bool = True,
    full_metrics: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    # Checkpoint: entrée
    try:
        logging.info(f"[SPAG_PLOT_START] band={band}, metric={metric}, rows={len(data)}")
    except Exception:
        pass
    plot_data = data.copy()
    if len(plot_data) == 0:
        return None

    plot_data['Subject_Base'] = plot_data['Subject'].astype(str).str.extract(r'(S\d+)')[0].fillna(plot_data['Subject'].astype(str))

    # Préparer source pour stats: per-epoch si possible
    stats_df: Optional[pd.DataFrame] = None
    if full_metrics is not None and len(full_metrics) > 0:
        stats_df = full_metrics.copy()
        stats_df['Subject_Base'] = stats_df['Subject'].astype(str).str.extract(r'(S\d+)')[0].fillna(stats_df['Subject'].astype(str))
    # Déterminer consensus par sujet
    consensus_map: Dict[str, Tuple[str, bool]] = {}
    subjects_unique = plot_data['Subject_Base'].unique()
    try:
        logging.info(f"[SPAG_PLOT_INFO] unique_subjects={len(subjects_unique)}")
    except Exception:
        pass
    for subject in subjects_unique:
        if stats_df is not None:
            filt = (stats_df['Subject_Base'] == subject) & (stats_df['Band'] == band)
            if 'Channel' in stats_df.columns:
                filt = filt & (stats_df['Channel'] == plot_data['Channel'].iloc[0])
            if 'stage' in stats_df.columns and 'stage' in plot_data.columns:
                filt = filt & (stats_df['stage'] == plot_data['stage'].iloc[0])
            sub_df = stats_df[filt]
        else:
            sub_df = plot_data[plot_data['Subject_Base'] == subject]
        x_b = sub_df[sub_df['Group'] == 'AVANT'][metric].dropna().to_numpy()
        x_a = sub_df[sub_df['Group'] == 'APRÈS'][metric].dropna().to_numpy()
        try:
            logging.info(f"[SPAG_PLOT_SUBJ] subj={subject}, n_before={x_b.size}, n_after={x_a.size}")
        except Exception:
            pass
        perm = permutation_test_median_diff(x_b, x_a)
        boot = bootstrap_ci_median_diff(x_b, x_a)
        rz = robust_z_intrasubject(x_b, x_a)
        decs = [perm['decision'], boot['decision'], rz['decision']]
        num_up = sum(d == 'augmentation' for d in decs)
        num_down = sum(d == 'diminution' for d in decs)
        if num_up >= 2:
            consensus_map[subject] = ('augmentation', True)
        elif num_down >= 2:
            consensus_map[subject] = ('diminution', True)
        else:
            consensus_map[subject] = ('stagnation', False)

    # Tracé
    fig, (ax, ax_table) = plt.subplots(1, 2, figsize=(20, 10), dpi=300, gridspec_kw={'width_ratios': [2, 1]})
    subjects = plot_data['Subject_Base'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, max(1, len(subjects))))
    subject_colors = dict(zip(subjects, colors))

    for subject in subjects:
        sd = plot_data[plot_data['Subject_Base'] == subject].sort_values('Group')
        x_positions = sd['Group'].map({'AVANT': 1, 'APRÈS': 2}).to_list()
        y_values = sd[metric].to_list()
        col = subject_colors[subject]
        is_sig = consensus_map.get(subject, ('stagnation', False))[1]
        marker = 'o'
        size = 130
        edge_w = 1.8
        # Main subject markers
        ax.scatter(x_positions, y_values, color=col, s=size, alpha=0.85, edgecolors='black', linewidth=edge_w, marker=marker, zorder=3)
        # Transparent overlay for additional depth
        ax.scatter(x_positions, y_values, color=col, s=60, alpha=0.28, edgecolors='none', linewidth=0, marker='o', zorder=4)
        if len(sd) == 2:
            ax.plot(x_positions, y_values, color=col, linewidth=2.5 if is_sig else 1.8, alpha=0.9 if is_sig else 0.7, zorder=2)
            if is_sig:
                # Étoile parfaitement centrée sur le point APRÈS
                ax.text(2, y_values[-1], '★', ha='center', va='center', color='gold', fontsize=18, fontweight='bold', zorder=6)


    # Axes
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['AVANT', 'APRÈS'], fontsize=12)
    channel = plot_data['Channel'].iloc[0] if 'Channel' in plot_data.columns else ''
    stage = plot_data['stage'].iloc[0] if 'stage' in plot_data.columns else 'ALL'
    try:
        logging.info(f"[SPAG_PLOT_AX] channel={channel}, stage={stage}")
    except Exception:
        pass
    ax.set_xlabel('Groupe', fontsize=14, fontweight='bold')
    if str(metric).upper() == 'AUC':
        ax.set_ylabel(f"AUC relative (0–1) — {band}", fontsize=14, fontweight='bold')
    else:
        ax.set_ylabel(f"{metric} (a.u.) — {band}", fontsize=14, fontweight='bold')
    ax.set_title(f"Canal {channel} — {band} — Stade {stage}{(' — ' + title_suffix) if title_suffix else ''}\n★: consensus (≥2/3)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.7, axis='y')
    ax.set_facecolor('#FAFAFA')

    # Legend inside plot (top-left), only if ≤ 20 subjects
    try:
        if len(subjects) <= 20:
            handles = [Line2D([0], [0], marker='o', color=subject_colors[s], label=str(s), markersize=4.5, linewidth=0) for s in subjects]
            leg = ax.legend(handles=handles, title='Sujets', loc='upper left', frameon=True, fontsize=8, ncol=2 if len(subjects) > 10 else 1)
            if leg and leg.get_frame():
                leg.get_frame().set_alpha(0.85)
                leg.get_frame().set_facecolor('white')
    except Exception:
        pass

    # P-value annotation (median across subjects)
    try:
        # Recompute p-values as in consensus pass
        pvals: List[float] = []
        subjects_loop = plot_data['Subject_Base'].unique()
        for subject in subjects_loop:
            if full_metrics is not None and len(full_metrics) > 0:
                filt = (full_metrics['Subject'].astype(str).str.contains(subject)) & (full_metrics['Band'] == band)
                if 'Channel' in full_metrics.columns:
                    filt = filt & (full_metrics['Channel'] == channel)
                if 'stage' in full_metrics.columns and 'stage' in plot_data.columns:
                    filt = filt & (full_metrics['stage'] == stage)
                sub_df2 = full_metrics[filt]
            else:
                sub_df2 = plot_data[plot_data['Subject_Base'] == subject]
            xb = sub_df2[sub_df2['Group'] == 'AVANT'][metric].dropna().to_numpy()
            xa = sub_df2[sub_df2['Group'] == 'APRÈS'][metric].dropna().to_numpy()
            pv = permutation_test_median_diff(xb, xa).get('p_value', np.nan)
            if not np.isnan(pv):
                pvals.append(float(pv))
        if pvals:
            p_med = float(np.median(pvals))
            txt = f"p = {p_med:.3f}"
            if p_med < 0.05:
                txt += "*"
            ymin, ymax = ax.get_ylim()
            ax.text(1.5, ymax - 0.02 * (ymax - ymin), txt, ha='center', va='top', fontsize=11, fontstyle='italic')
    except Exception:
        pass

    # Tableau latéral
    ax_table.axis('off')
    headers, rows, style_meta = _build_table_rows(plot_data, metric=metric, band=band, stats_df=stats_df)
    table_data = [headers] + rows
    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.09, 0.09, 0.09, 0.11, 0.11, 0.09, 0.11, 0.14, 0.07, 0.09, 0.09, 0.09, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1, 1.45)
    # Header style
    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_facecolor('#455A64')
        cell.set_text_props(weight='bold', color='white')
    # Décisions colorées
    for i, meta in enumerate(style_meta, start=1):
        for col_idx, key in [(9, 'perm'), (10, 'boot'), (11, 'z'), (12, 'cons')]:
            cell = table[(i, col_idx)]
            cell.set_facecolor(_color_for_decision(meta[key]))

    ax_table.set_title('Intra-sujet (robuste) — flèches: ↑/↓/=', fontsize=11, fontweight='bold', pad=14)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filename = f"Spaghetti_{channel}_{stage}_{band}_{metric}.png"
    filepath = os.path.join(output_dir, filename)
    try:
        logging.info(f"[SPAG_PLOT_SAVE] path={filepath}")
    except Exception:
        pass
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Export CSV des données utilisées (table)
    if export_csv and len(rows) > 0:
        csv_headers = headers
        df_out = pd.DataFrame(rows, columns=csv_headers)
        csv_path = os.path.join(output_dir, f"Spaghetti_{channel}_{stage}_{band}_{metric}_table.csv")
        try:
            df_out.to_csv(csv_path, index=False, encoding='utf-8')
            try:
                logging.info(f"[SPAG_PLOT_CSV] path={csv_path}")
            except Exception:
                pass
        except Exception:
            pass

    return filepath


# =====================
# API de génération
# =====================
def _filter_metrics(
    metrics: pd.DataFrame,
    selected_bands: Optional[Iterable[str]] = None,
    selected_stages: Optional[Iterable[str]] = None,
    selected_channels: Optional[Iterable[str]] = None,
    selected_subjects: Optional[Iterable[str]] = None,
    selected_band_stage_map: Optional[Dict[str, Iterable[str]]] = None,
) -> pd.DataFrame:
    df = metrics.copy()
    if selected_bands:
        df = df[df['Band'].isin(list(selected_bands))]
    if selected_stages and 'stage' in df.columns:
        df = df[df['stage'].isin(list(selected_stages))]
    if selected_band_stage_map and 'stage' in df.columns:
        # Garder seulement les lignes dont (Band, stage) figure dans la carte
        allowed = []
        for band, stages in selected_band_stage_map.items():
            for st in stages:
                allowed.append((band, st))
        if allowed:
            allowed_set = set(allowed)
            df = df[df.apply(lambda r: (str(r['Band']), str(r['stage'])) in allowed_set, axis=1)]
    if selected_channels:
        df = df[df['Channel'].isin(list(selected_channels))]
    if selected_subjects:
        # Correspondance souple via préfixe Sxxx
        bases = pd.Series(list(selected_subjects)).astype(str).str.extract(r'(S\d+)')[0].dropna().unique().tolist()
        df['Subject_Base'] = df['Subject'].astype(str).str.extract(r'(S\d+)')[0]
        df = df[df['Subject_Base'].isin(bases)]
        df = df.drop(columns=['Subject_Base'])
    return df


def _make_combinations(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    bands = sorted(df['Band'].unique())
    channels = sorted(df['Channel'].unique())
    stages = sorted(df['stage'].unique()) if 'stage' in df.columns else ['ALL']
    combos: List[Tuple[str, str, str]] = []
    for ch in channels:
        for st in stages:
            for b in bands:
                combos.append((ch, st, b))
    return combos


def _aggregate_for_plot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # Agréger par sujet (median intra-sujet par groupe) pour le spaghetti
    df = df.copy()
    df['Subject_Base'] = df['Subject'].astype(str).str.extract(r'(S\d+)')[0].fillna(df['Subject'].astype(str))
    agg = df.groupby(['Subject_Base', 'Subject', 'Channel', 'Group', 'stage', 'Band'])[metric].median().reset_index()
    return agg


def generate_spaghetti_from_dataframe(
    df_fft: pd.DataFrame,
    output_dir: str,
    selected_bands: Optional[Iterable[str]] = None,
    selected_stages: Optional[Iterable[str]] = None,
    selected_channels: Optional[Iterable[str]] = None,
    selected_subjects: Optional[Iterable[str]] = None,
    bands_mapping: Optional[Dict[str, Tuple[float, float]]] = None,
    total_range: Tuple[float, float] = (0.5, 100.0),
    metric: str = 'AUC',
) -> List[str]:
    # Calcul des métriques à partir des FFT
    metrics = compute_band_metrics_from_fft(df_fft, bands=bands_mapping or DEFAULT_BANDS, total_range=total_range)
    # Filtrage
    metrics = _filter_metrics(metrics, selected_bands, selected_stages, selected_channels, selected_subjects)
    if len(metrics) == 0:
        return []
    # Préparer données pour graphes
    agg = _aggregate_for_plot(metrics, metric=metric)
    outputs: List[str] = []
    for ch, st, b in _make_combinations(agg):
        sub = agg[(agg['Channel'] == ch) & (agg['Band'] == b) & (agg['stage'] == st)]
        if len(sub) == 0:
            continue
        # Passer les valeurs par epoch du même band/canal/stage comme source pour les stats
        stats_src = metrics[(metrics['Channel'] == ch) & (metrics['Band'] == b) & (metrics['stage'] == st)]
        out = create_spaghetti_plot(sub, metric=metric, band=b, output_dir=os.path.join(output_dir, f"{ch}"), title_suffix=f"Stage {st}", full_metrics=stats_src)
        if out:
            outputs.append(out)
    return outputs


def generate_spaghetti_from_fft_csv(
    before_dir: str,
    after_dir: str,
    output_dir: str,
    selected_bands: Optional[Iterable[str]] = None,
    selected_stages: Optional[Iterable[str]] = None,
    selected_channels: Optional[Iterable[str]] = None,
    selected_subjects: Optional[Iterable[str]] = None,
    bands_mapping: Optional[Dict[str, Tuple[float, float]]] = None,
    total_range: Tuple[float, float] = (0.5, 45.0),
    metric: str = 'AUC',
) -> List[str]:
    # Charger FFT CSV
    df_b = _load_fft_group(before_dir, 'AVANT')
    df_a = _load_fft_group(after_dir, 'APRÈS')
    df = pd.concat([df_b, df_a], ignore_index=True)
    return generate_spaghetti_from_dataframe(
        df_fft=df,
        output_dir=output_dir,
        selected_bands=selected_bands,
        selected_stages=selected_stages,
        selected_channels=selected_channels,
        selected_subjects=selected_subjects,
        bands_mapping=bands_mapping,
        total_range=total_range,
        metric=metric,
    )


__all__ = [
    'generate_spaghetti_from_fft_csv',
    'generate_spaghetti_from_dataframe',
    'compute_band_metrics_from_fft',
    'permutation_test_median_diff',
    'bootstrap_ci_median_diff',
    'robust_z_intrasubject',
    'generate_spaghetti_from_edf_dirs',
    'generate_spaghetti_from_edf_file_lists',
]


# =====================
# EDF -> métriques de bande par epoch
# =====================
def _canonical_stage(stage: str) -> Optional[str]:
    if stage is None:
        return None
    s = str(stage).strip().upper()
    mapping = {
        'W': 'W', 'WAKE': 'W', 'AWAKE': 'W', 'EVEIL': 'W', 'ÉVEIL': 'W',
        'R': 'R', 'REM': 'R', 'PARADOXAL': 'R', 'SOMMEIL PARADOXAL': 'R',
        'N1': 'N1', 'S1': 'N1', 'NREM1': 'N1', 'STAGE1': 'N1', 'STAGE 1': 'N1',
        'N2': 'N2', 'S2': 'N2', 'NREM2': 'N2', 'STAGE2': 'N2', 'STAGE 2': 'N2',
        'N3': 'N3', 'S3': 'N3', 'S4': 'N3', 'NREM3': 'N3', 'NREM4': 'N3', 'STAGE3': 'N3', 'STAGE 3': 'N3', 'STAGE4': 'N3', 'STAGE 4': 'N3',
    }
    return mapping.get(s, None)


def _find_hypnogram_path(psg_path: str) -> Optional[str]:
    directory = os.path.dirname(psg_path)
    # Heuristique simple: chercher fichiers contenant "Hypnogram" dans le dossier
    cands = glob.glob(os.path.join(directory, "*Hypnogram*.edf"))
    if cands:
        return cands[0]
    return None


def _extract_epochs(signal: np.ndarray, fs: float, scoring_df: Optional[pd.DataFrame], epoch_len: float = 30.0,
                    required_stages: Optional[Iterable[str]] = None) -> List[Tuple[str, np.ndarray]]:
    n = len(signal)
    epochs: List[Tuple[str, np.ndarray]] = []
    if scoring_df is None or len(scoring_df) == 0:
        # Sans scoring: étiqueter tout en W
        step = int(round(epoch_len * fs))
        for i0 in range(0, n, step):
            seg = signal[i0:min(n, i0 + step)]
            if seg.size >= int(fs):
                epochs.append(('W', seg))
        return epochs
    eplen = float(epoch_len)
    # Filtrer les stades si demandé
    allowed: Optional[set] = set([str(s).upper() for s in required_stages]) if required_stages else None
    for _, row in scoring_df.iterrows():
        st = _canonical_stage(row.get('stage', ''))
        if st is None:
            continue
        if allowed is not None and st not in allowed:
            continue
        t0 = float(row.get('time', 0.0))
        i0 = int(max(0, min(n - 1, round(t0 * fs))))
        i1 = int(max(i0 + 1, min(n, round((t0 + eplen) * fs))))
        seg = signal[i0:i1]
        if seg.size >= int(fs):  # >= 1s
            epochs.append((st, seg))
    return epochs


def _band_auc_from_epoch(seg: np.ndarray, fs: float, bands: Dict[str, Tuple[float, float]], total_range: Tuple[float, float]) -> Dict[str, Dict[str, float]]:
    # FFT magnitude par epoch
    x = seg.astype(float)
    x = x - np.mean(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(x))
    # Restreindre à la fenêtre totale
    mask_total = (freqs >= total_range[0]) & (freqs <= total_range[1])
    freqs_t = freqs[mask_total]
    spec_t = spec[mask_total]
    eps = np.finfo(float).eps
    total_auc = float(np.trapz(spec_t, freqs_t)) + eps
    out: Dict[str, Dict[str, float]] = {}
    for name, (lo, hi) in bands.items():
        m = (freqs >= lo) & (freqs <= hi)
        f = freqs[m]
        s = spec[m]
        if f.size == 0:
            out[name] = {
                'AUC': np.nan,
                'Peak_Amplitude': np.nan,
                'Peak_Frequency': np.nan,
                'Barycenter': np.nan,
            }
            continue
        auc_b = float(np.trapz(s, f))
        rel = auc_b / total_auc if total_auc > 0 else np.nan
        peak_amp = float(s.max())
        peak_freq = float(f[np.argmax(s)])
        denom = float(s.sum())
        bary = float((f * s).sum() / denom) if denom > 0 else np.nan
        out[name] = {
            'AUC': rel,
            'Peak_Amplitude': peak_amp,
            'Peak_Frequency': peak_freq,
            'Barycenter': bary,
        }
    return out


def _edf_to_metrics(
    edf_files: List[str],
    group_name: str,
    selected_channels: Optional[Iterable[str]],
    bands_mapping: Dict[str, Tuple[float, float]],
    epoch_len: float,
    total_range: Tuple[float, float],
    edf_to_excel_map: Optional[Dict[str, str]] = None,
    required_stages: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if mne is None:
        raise ImportError("mne doit être installé pour charger des fichiers EDF")
    for psg_path in edf_files:
        t_file_start = time.time()
        try:
            logging.info(f"[EDF_FILE] path='{psg_path}', group='{group_name}'")
        except Exception:
            pass
        try:
            # Lecture paresseuse pour éviter de charger 80 canaux si un seul est requis
            raw = mne.io.read_raw_edf(psg_path, preload=False, verbose='ERROR')  # type: ignore
        except Exception:
            try:
                logging.info(f"[EDF_ERROR] read_raw failed: {psg_path}")
            except Exception:
                pass
            continue
        fs = float(raw.info.get('sfreq', 0.0))
        if not fs or fs <= 0:
            try:
                logging.info(f"[EDF_SKIP] invalid fs={fs} for {psg_path}")
            except Exception:
                pass
            continue
        ch_names = list(raw.info.get('ch_names', []))
        if selected_channels:
            keep = [c for c in ch_names if c in set(selected_channels)]
        else:
            # Limiter aux types EEG si possible
            try:
                picks = mne.pick_types(raw.info, eeg=True, eog=False, emg=False)  # type: ignore
                keep = [ch_names[i] for i in picks]
            except Exception:
                keep = ch_names
        # Réduire le Raw aux seuls canaux nécessaires pour accélérer les accès
        try:
            if keep and hasattr(raw, 'pick'):
                raw.pick(keep)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            logging.info(f"[EDF_LOADED] fs={fs}, ch_total={len(ch_names)}, ch_kept={len(keep)}")
        except Exception:
            pass

        # Hypnogram
        scoring_df: Optional[pd.DataFrame] = None
        if scoring_io is not None:
            ann_path = _find_hypnogram_path(psg_path)
            try:
                if ann_path:
                    dur_s = float(raw.times[-1]) if hasattr(raw, 'times') else float(raw.n_times / raw.info['sfreq'])
                    scoring_df = scoring_io.import_edf_hypnogram(ann_path, recording_duration_s=dur_s)  # type: ignore
            except Exception:
                scoring_df = None

        # Fallback: Excel scoring avec même nom de base (comme dans le batch)
        if scoring_df is None and scoring_io is not None:
            # 1) Utiliser mapping fourni depuis l'UI batch si dispo
            excel_path = None
            if edf_to_excel_map is not None:
                # Essayer clé telle quelle puis normalisée
                excel_path = edf_to_excel_map.get(psg_path)
                if excel_path is None:
                    try:
                        excel_path = edf_to_excel_map.get(os.path.abspath(psg_path))
                    except Exception:
                        pass
            # 2) Sinon, chercher fichier Excel à côté de l'EDF
            if excel_path is None:
                base_no_ext = os.path.splitext(psg_path)[0]
                for ext in ('.xlsx', '.xls'):
                    cand = base_no_ext + ext
                    if os.path.exists(cand):
                        excel_path = cand
                        break
            if excel_path and os.path.exists(excel_path):
                try:
                    xdf = pd.read_excel(excel_path)
                    scoring_df = scoring_io.import_excel_scoring(xdf, absolute_start_datetime=None, epoch_seconds=30.0)  # type: ignore
                except Exception:
                    scoring_df = None
        try:
            n_sc = 0 if scoring_df is None else len(scoring_df)
            logging.info(f"[SCORING] source={'edf' if scoring_df is not None else 'none/excel_fail'}, rows={n_sc}")
        except Exception:
            pass

        subj = Path(psg_path).stem
        for ch in keep:
            try:
                logging.info(f"[EDF_CHANNEL] subj='{subj}', ch='{ch}'")
            except Exception:
                pass
            try:
                data, _ = raw.get_data(picks=[ch], return_times=True)  # type: ignore
            except Exception:
                try:
                    logging.info(f"[EDF_ERROR] get_data failed for channel {ch}")
                except Exception:
                    pass
                continue
            sig = np.asarray(data[0], dtype=float)
            epochs = _extract_epochs(sig, fs=fs, scoring_df=scoring_df, epoch_len=epoch_len, required_stages=required_stages)
            try:
                logging.info(f"[EPOCHS] ch='{ch}', n_epochs={len(epochs)}")
            except Exception:
                pass
            epoch_idx = 0
            for st, seg in epochs:
                if epoch_idx == 0 or (epoch_idx % 50 == 0):
                    try:
                        logging.info(f"[EPOCH_PROC] ch='{ch}', epoch={epoch_idx}/{len(epochs)}, stage={st}")
                    except Exception:
                        pass
                metrics = _band_auc_from_epoch(seg, fs=fs, bands=bands_mapping, total_range=total_range)
                for band_name, vals in metrics.items():
                    rows.append({
                        'Subject': subj,
                        'Channel': ch,
                        'Group': group_name,
                        'stage': st,
                        'Epoch': epoch_idx,
                        'Band': band_name,
                        'AUC': vals['AUC'],
                        'Peak_Amplitude': vals['Peak_Amplitude'],
                        'Peak_Frequency': vals['Peak_Frequency'],
                        'Barycenter': vals['Barycenter'],
                    })
                epoch_idx += 1
            try:
                logging.info(f"[CHANNEL_DONE] subj='{subj}', ch='{ch}', rows_added={epoch_idx * len(bands_mapping)}")
            except Exception:
                pass
        try:
            logging.info(f"[FILE_DONE] path='{psg_path}', rows_total={len(rows)}, elapsed_s={time.time()-t_file_start:.2f}")
        except Exception:
            pass
    return pd.DataFrame(rows)


def generate_spaghetti_from_edf_dirs(
    before_dir: str,
    after_dir: str,
    output_dir: str,
    selected_bands: Optional[Iterable[str]] = None,
    selected_stages: Optional[Iterable[str]] = None,
    selected_channels: Optional[Iterable[str]] = None,
    selected_subjects: Optional[Iterable[str]] = None,
    selected_band_stage_map: Optional[Dict[str, Iterable[str]]] = None,
    bands_mapping: Optional[Dict[str, Tuple[float, float]]] = None,
    total_range: Tuple[float, float] = (0.5, 45.0),
    epoch_len: float = 30.0,
    metric: str = 'AUC',
    rng_seed: int = RNG_SEED,
    n_perm: int = N_PERM,
    n_boot: int = N_BOOT,
    edf_to_excel_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Génère des graphiques spaghetti directement depuis des dossiers contenant des EDF.

    Les dossiers `before_dir` et `after_dir` sont interprétés comme Group=AVANT et APRÈS.
    """
    # Config RNG globale (réproducibilité sur méthodes)
    global RNG_SEED, N_PERM, N_BOOT
    RNG_SEED = int(rng_seed)
    N_PERM = int(n_perm)
    N_BOOT = int(n_boot)

    mapping = bands_mapping or DEFAULT_BANDS
    try:
        logging.info(f"[SPAG_DIRS_START] before_dir='{before_dir}', after_dir='{after_dir}', out='{output_dir}'")
    except Exception:
        pass
    # Détection fichiers EDF comme dans l'automatisation FFT (extensions multiples)
    def _scan_edf(dir_path: str) -> List[str]:
        hits: List[str] = []
        exts = ['.edf', '.EDF', '.edf+', '.EDF+']
        for root, _dirs, files in os.walk(dir_path):
            for f in files:
                if any(f.endswith(ext) for ext in exts):
                    hits.append(os.path.join(root, f))
        return hits

    edf_before = _scan_edf(before_dir)
    edf_after = _scan_edf(after_dir)
    try:
        logging.info(f"[SPAG_DIRS_FILES] before={len(edf_before)}, after={len(edf_after)}")
    except Exception:
        pass
    if not edf_before and not edf_after:
        return []

    print(f"Chargement EDF AVANT: {len(edf_before)} fichiers; APRÈS: {len(edf_after)} fichiers")
    # Calculer l'union des stades requis depuis la carte bande->stades
    req_stages = None
    if selected_band_stage_map:
        req_set = set()
        for stages in selected_band_stage_map.values():
            for st in stages:
                req_set.add(st)
        req_stages = sorted(req_set)

    df_b = _edf_to_metrics(edf_before, 'AVANT', selected_channels, mapping, epoch_len, total_range, edf_to_excel_map, req_stages) if edf_before else pd.DataFrame()
    df_a = _edf_to_metrics(edf_after, 'APRÈS', selected_channels, mapping, epoch_len, total_range, edf_to_excel_map, req_stages) if edf_after else pd.DataFrame()
    metrics = pd.concat([df_b, df_a], ignore_index=True)
    try:
        logging.info(f"[SPAG_DIRS_METRICS] total_rows={len(metrics)}")
    except Exception:
        pass

    # Filtres
    metrics = _filter_metrics(metrics, selected_bands, selected_stages, selected_channels, selected_subjects, selected_band_stage_map)
    if len(metrics) == 0:
        print("Aucune donnée après filtrage.")
        return []

    # Agrégation pour le spaghetti (médiane intra-sujet par groupe)
    agg = _aggregate_for_plot(metrics, metric=metric)
    try:
        logging.info(f"[SPAG_DIRS_AGG] combos={len(_make_combinations(agg))}")
    except Exception:
        pass

    # Génération des graphiques par combinaison Canal × Stade × Bande
    outputs: List[str] = []
    decisions_counter = {'augmentation': 0, 'diminution': 0, 'stagnation': 0}
    for ch, st, b in _make_combinations(agg):
        sub = agg[(agg['Channel'] == ch) & (agg['Band'] == b) & (agg['stage'] == st)]
        try:
            logging.info(f"[SPAG_DIRS_COMBO] ch={ch}, stage={st}, band={b}, n={len(sub)}")
        except Exception:
            pass
        if len(sub) == 0:
            continue
        stats_src = metrics[(metrics['Channel'] == ch) & (metrics['Band'] == b) & (metrics['stage'] == st)]
        out = create_spaghetti_plot(
            sub,
            metric=metric,
            band=b,
            output_dir=os.path.join(output_dir, f"{ch}"),
            title_suffix=f"Stage {st}",
            full_metrics=stats_src,
        )
        if out:
            outputs.append(out)
            # Résumé rapide: compter décisions consensus
            # Déterminer consensus global sur ce combo (majorité des sujets)
            sub_stats = []
            for subj in sub['Subject'].astype(str).str.extract(r'(S\d+)')[0].unique():
                ssrc = stats_src.copy()
                ssrc['Subject_Base'] = ssrc['Subject'].astype(str).str.extract(r'(S\d+)')[0]
                ssrc = ssrc[ssrc['Subject_Base'] == subj]
                x_b = ssrc[ssrc['Group'] == 'AVANT'][metric].dropna().to_numpy()
                x_a = ssrc[ssrc['Group'] == 'APRÈS'][metric].dropna().to_numpy()
                perm = permutation_test_median_diff(x_b, x_a)
                boot = bootstrap_ci_median_diff(x_b, x_a)
                rz = robust_z_intrasubject(x_b, x_a)
                decs = [perm['decision'], boot['decision'], rz['decision']]
                num_up = sum(d == 'augmentation' for d in decs)
                num_down = sum(d == 'diminution' for d in decs)
                if num_up >= 2:
                    sub_stats.append('augmentation')
                elif num_down >= 2:
                    sub_stats.append('diminution')
                else:
                    sub_stats.append('stagnation')
            if sub_stats:
                # Majorité simple
                maj = max(set(sub_stats), key=sub_stats.count)
                decisions_counter[maj] += 1

    print("\n=== RÉSUMÉ GLOBAL (consensus par combinaison) ===")
    for k, v in decisions_counter.items():
        print(f"  {k}: {v}")
    try:
        logging.info(f"[SPAG_DIRS_SUMMARY] {decisions_counter}")
    except Exception:
        pass
    return outputs


def generate_spaghetti_from_edf_file_lists(
    before_files: List[str],
    after_files: List[str],
    output_dir: str,
    selected_bands: Optional[Iterable[str]] = None,
    selected_stages: Optional[Iterable[str]] = None,
    selected_channels: Optional[Iterable[str]] = None,
    selected_subjects: Optional[Iterable[str]] = None,
    selected_band_stage_map: Optional[Dict[str, Iterable[str]]] = None,
    bands_mapping: Optional[Dict[str, Tuple[float, float]]] = None,
    total_range: Tuple[float, float] = (0.5, 45.0),
    epoch_len: float = 30.0,
    metric: str = 'AUC',
    rng_seed: int = RNG_SEED,
    n_perm: int = N_PERM,
    n_boot: int = N_BOOT,
    edf_to_excel_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    global RNG_SEED, N_PERM, N_BOOT
    RNG_SEED = int(rng_seed)
    N_PERM = int(n_perm)
    N_BOOT = int(n_boot)
    mapping = bands_mapping or DEFAULT_BANDS
    try:
        logging.info(f"[SPAG_LIST_START] before={len(before_files)}, after={len(after_files)}, out='{output_dir}'")
    except Exception:
        pass
    req_stages = None
    if selected_band_stage_map:
        req_set = set()
        for stages in selected_band_stage_map.values():
            for st in stages:
                req_set.add(st)
        req_stages = sorted(req_set)

    df_b = _edf_to_metrics(before_files, 'AVANT', selected_channels, mapping, epoch_len, total_range, edf_to_excel_map, req_stages) if before_files else pd.DataFrame()
    df_a = _edf_to_metrics(after_files, 'APRÈS', selected_channels, mapping, epoch_len, total_range, edf_to_excel_map, req_stages) if after_files else pd.DataFrame()
    metrics = pd.concat([df_b, df_a], ignore_index=True)
    try:
        logging.info(f"[SPAG_LIST_METRICS] total_rows={len(metrics)}")
    except Exception:
        pass
    metrics = _filter_metrics(metrics, selected_bands, selected_stages, selected_channels, selected_subjects, selected_band_stage_map)
    if len(metrics) == 0:
        return []
    agg = _aggregate_for_plot(metrics, metric=metric)
    try:
        logging.info(f"[SPAG_LIST_AGG] combos={len(_make_combinations(agg))}")
    except Exception:
        pass
    outputs: List[str] = []
    for ch, st, b in _make_combinations(agg):
        sub = agg[(agg['Channel'] == ch) & (agg['Band'] == b) & (agg['stage'] == st)]
        try:
            logging.info(f"[SPAG_LIST_COMBO] ch={ch}, stage={st}, band={b}, n={len(sub)}")
        except Exception:
            pass
        if len(sub) == 0:
            continue
        stats_src = metrics[(metrics['Channel'] == ch) & (metrics['Band'] == b) & (metrics['stage'] == st)]
        out = create_spaghetti_plot(sub, metric=metric, band=b, output_dir=os.path.join(output_dir, f"{ch}"), title_suffix=f"Stage {st}", full_metrics=stats_src)
        if out:
            outputs.append(out)
    return outputs


