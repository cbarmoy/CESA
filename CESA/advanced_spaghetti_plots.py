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
- Consensus ≥ 2/3: annotation "*" sur le graphique
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import logging
import time
from scipy import stats as scipy_stats
 
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


def _build_table_rows(
    plot_data: pd.DataFrame, 
    metric: str, 
    band: str, 
    stats_df: Optional[pd.DataFrame],
    subject_colors: Optional[Dict[Any, Tuple[float, float, float]]] = None
) -> Tuple[List[str], List[List[str]], List[Dict[str, Any]]]:
    headers = [
        'Subject', 'Band', 'Channel', 'Median Before', 'Median After', 'Diff',
        'p (Perm.)', 'CI (Boot)', 'Z',
        'Perm', 'Boot', 'Zsc', 'Sig', 'Consensus'
    ]
    subjects = plot_data['Subject_Base'].unique().tolist()
    channel = plot_data['Channel'].iloc[0] if 'Channel' in plot_data.columns else ''
    table_rows: List[List[str]] = []
    style_meta: List[Dict[str, Any]] = []
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
            cons_text = 'UP'
            sig_flag = '*'
        elif num_down >= 2:
            cons_dec = 'diminution'
            cons_text = 'DN'
            sig_flag = '*'
        else:
            cons_dec = 'stagnation' if all(d == 'stagnation' for d in decs) else 'mixte'
            cons_text = '=' if cons_dec == 'stagnation' else '~'
            sig_flag = ''
        
        def _sym(decision: str) -> str:
            d = str(decision)
            if d == 'augmentation':
                return 'UP'
            if d == 'diminution':
                return 'DN'
            if d == 'mixte':
                return '~'
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
            sig_flag,
            cons_text,
        ]
        table_rows.append(row)
        # Include subject color in style_meta
        subj_color = subject_colors.get(subj) if subject_colors else None
        style_meta.append({
            'perm': str(perm['decision']), 
            'boot': str(boot['decision']), 
            'z': str(rz['decision']), 
            'cons': cons_dec, 
            'sig': sig_flag,
            'subject_color': subj_color
        })
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
    clusters: Optional[Dict[str, str]] = None,
    cluster_names: Optional[Dict[str, str]] = None,
    before_label: str = "Before",
    after_label: str = "After",
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

    # Compute global and per-cluster means with Wilcoxon tests
    # Collect before/after values per subject
    subject_pairs: Dict[str, Tuple[float, float]] = {}
    for subject in subjects_unique:
        sd = plot_data[plot_data['Subject_Base'] == subject].sort_values('Group')
        groups = sd['Group'].unique()
        if len(groups) == 2 and 'AVANT' in [str(g).upper() for g in groups]:
            before_val = sd[sd['Group'].str.upper() == 'AVANT'][metric].values
            after_val = sd[sd['Group'].str.upper() == 'APRÈS'][metric].values
            if len(before_val) > 0 and len(after_val) > 0:
                subject_pairs[subject] = (float(before_val[0]), float(after_val[0]))

    # Global statistics
    global_stats = {'mean_before': np.nan, 'mean_after': np.nan, 'p_value': np.nan, 'label': 'Global Mean'}
    if len(subject_pairs) >= 2:
        before_vals = np.array([v[0] for v in subject_pairs.values()])
        after_vals = np.array([v[1] for v in subject_pairs.values()])
        global_stats['mean_before'] = float(np.mean(before_vals))
        global_stats['mean_after'] = float(np.mean(after_vals))
        try:
            stat, pval = scipy_stats.wilcoxon(before_vals, after_vals, alternative='two-sided')
            global_stats['p_value'] = float(pval)
        except Exception:
            global_stats['p_value'] = np.nan

    # Per-cluster statistics
    cluster_stats: Dict[str, Dict[str, Any]] = {}
    normalized_clusters_temp: Dict[str, str] = {}
    if clusters:
        for key, cid in clusters.items():
            if key is None or cid is None:
                continue
            k = str(key).strip().upper()
            if not k:
                continue
            normalized_clusters_temp[k] = str(cid).strip().upper() or 'A'
    
    if normalized_clusters_temp:
        cluster_pairs: Dict[str, List[Tuple[float, float]]] = {}
        for subject, (bef, aft) in subject_pairs.items():
            subj_key = str(subject).strip().upper()
            cid = normalized_clusters_temp.get(subj_key, 'UNASSIGNED')
            if cid != 'UNASSIGNED':
                cluster_pairs.setdefault(cid, []).append((bef, aft))
        
        for cid, pairs in cluster_pairs.items():
            if len(pairs) >= 2:
                before_vals = np.array([p[0] for p in pairs])
                after_vals = np.array([p[1] for p in pairs])
                mean_before = float(np.mean(before_vals))
                mean_after = float(np.mean(after_vals))
                try:
                    stat, pval = scipy_stats.wilcoxon(before_vals, after_vals, alternative='two-sided')
                    p_value = float(pval)
                except Exception:
                    p_value = np.nan
                cluster_label = f"Cluster {cid}"
                if cluster_names and cid in cluster_names:
                    cluster_label = str(cluster_names[cid])
                cluster_stats[cid] = {
                    'mean_before': mean_before,
                    'mean_after': mean_after,
                    'p_value': p_value,
                    'label': cluster_label
                }

    # Tracé
    fig, (ax, ax_table) = plt.subplots(
        2,
        1,
        figsize=(14, 12),
        dpi=300,
        gridspec_kw={'height_ratios': [2.2, 1.0]},
    )
    subjects = plot_data['Subject_Base'].unique()
    subject_colors: Dict[Any, Tuple[float, float, float]] = {}
    cluster_subjects: Dict[str, List[Any]] = {}
    cluster_label_map: Dict[str, str] = {}
    cluster_rep_colors: Dict[str, Tuple[float, float, float]] = {}

    def _palette_for_cluster(cluster_id: str, size: int) -> List[Tuple[float, float, float]]:
        if size <= 0:
            return []
        cid = str(cluster_id).upper()
        if cid == 'A':
            # Highly saturated red-orange palette with MAXIMUM differentiation
            if size == 1:
                return [(0.95, 0.15, 0.0)]  # Vivid red-orange
            elif size == 2:
                return [(0.85, 0.05, 0.05), (1.0, 0.50, 0.0)]  # Deep red to bright orange
            elif size == 3:
                return [(0.75, 0.0, 0.0), (0.95, 0.25, 0.0), (1.0, 0.55, 0.0)]  # Dark red, red-orange, orange
            elif size == 4:
                return [(0.65, 0.0, 0.0), (0.85, 0.10, 0.0), (0.95, 0.35, 0.0), (1.0, 0.60, 0.0)]
            elif size == 5:
                return [(0.60, 0.0, 0.0), (0.80, 0.05, 0.0), (0.95, 0.25, 0.0), (1.0, 0.45, 0.0), (1.0, 0.65, 0.0)]
            else:
                # For larger sets, create evenly spaced highly saturated colors
                colors = []
                for i in range(size):
                    # Vary hue from deep red (0.6,0,0) to bright orange (1.0,0.7,0)
                    ratio = i / max(1, size - 1)
                    r = 0.60 + 0.40 * ratio
                    g = 0.0 + 0.70 * ratio
                    b = 0.0
                    colors.append((r, g, b))
                return colors
        if cid == 'B':
            # Highly saturated blue palette with MAXIMUM differentiation
            if size == 1:
                return [(0.0, 0.20, 0.95)]  # Vivid blue
            elif size == 2:
                return [(0.0, 0.10, 0.75), (0.0, 0.45, 1.0)]  # Deep blue to bright blue
            elif size == 3:
                return [(0.0, 0.05, 0.65), (0.0, 0.25, 0.85), (0.0, 0.50, 1.0)]  # Dark blue, medium blue, bright blue
            elif size == 4:
                return [(0.0, 0.0, 0.60), (0.0, 0.15, 0.75), (0.0, 0.35, 0.90), (0.0, 0.55, 1.0)]
            elif size == 5:
                return [(0.0, 0.0, 0.55), (0.0, 0.10, 0.70), (0.0, 0.25, 0.85), (0.0, 0.45, 0.95), (0.0, 0.60, 1.0)]
            else:
                # For larger sets, create evenly spaced highly saturated blues
                colors = []
                for i in range(size):
                    # Vary brightness from dark blue (0,0,0.55) to bright blue (0,0.65,1.0)
                    ratio = i / max(1, size - 1)
                    r = 0.0
                    g = 0.0 + 0.65 * ratio
                    b = 0.55 + 0.45 * ratio
                    colors.append((r, g, b))
                return colors
        if cid == 'UNASSIGNED':
            return [(0.7, 0.7, 0.7)] * size
        return list(sns.color_palette("husl", size))

    normalized_clusters: Dict[str, str] = {}
    if clusters:
        for key, cid in clusters.items():
            if key is None or cid is None:
                continue
            k = str(key).strip().upper()
            if not k:
                continue
            normalized_clusters[k] = str(cid).strip().upper() or 'A'
    if cluster_names:
        for cid, name in cluster_names.items():
            if name is None:
                continue
            cid_norm = str(cid).strip().upper()
            label = str(name).strip()
            if cid_norm and label:
                cluster_label_map[cid_norm] = label
    for default_id in ['A', 'B']:
        cluster_label_map.setdefault(default_id, f"Cluster {default_id}")

    if not normalized_clusters:
        palette = plt.cm.viridis(np.linspace(0, 1, max(1, len(subjects))))
        subject_colors = dict(zip(subjects, palette))
        cluster_subjects['DEFAULT'] = list(subjects)
        cluster_label_map.setdefault('DEFAULT', 'Subjects')
        cluster_order = ['DEFAULT']
    else:
        for subject in subjects:
            subj_key = str(subject).strip().upper()
            cid = normalized_clusters.get(subj_key, 'UNASSIGNED')
            cluster_subjects.setdefault(cid, []).append(subject)
        if 'UNASSIGNED' in cluster_subjects:
            cluster_label_map.setdefault('UNASSIGNED', 'Unassigned')
        cluster_order: List[str] = []
        for preferred in ['A', 'B']:
            if preferred in cluster_subjects:
                cluster_order.append(preferred)
        for cid in sorted(cluster_subjects.keys()):
            if cid not in cluster_order:
                cluster_order.append(cid)
        for cid in cluster_order:
            members = sorted(cluster_subjects.get(cid, []), key=lambda s: str(s))
            cluster_subjects[cid] = members
            palette = _palette_for_cluster(cid, len(members))
            if not palette:
                palette = [(0.3, 0.3, 0.3)] * len(members)
            for subj, color in zip(members, palette):
                subject_colors[subj] = color
            cluster_rep_colors[cid] = palette[min(len(palette) - 1, max(0, len(palette) // 2))]
    if 'cluster_order' not in locals():
        cluster_order = ['DEFAULT']

    group_to_pos = {
        'AVANT': 1,
        'APRÈS': 2,
        'APRES': 2,
        'BEFORE': 1,
        'AFTER': 2,
    }

    for subject in subjects:
        sd = plot_data[plot_data['Subject_Base'] == subject].sort_values('Group')
        x_positions = sd['Group'].apply(lambda g: group_to_pos.get(str(g).upper(), 1)).to_list()
        y_values = sd[metric].to_list()
        col = subject_colors[subject]
        is_sig = consensus_map.get(subject, ('stagnation', False))[1]
        marker = 'o'
        size = 200
        edge_w = 2.8
        # Main subject markers
        ax.scatter(x_positions, y_values, color=col, s=size, alpha=0.85, edgecolors='black', linewidth=edge_w, marker=marker, zorder=3)
        # Transparent overlay for additional depth
        ax.scatter(x_positions, y_values, color=col, s=110, alpha=0.3, edgecolors='none', linewidth=0, marker='o', zorder=4)
        if len(sd) == 2:
            ax.plot(x_positions, y_values, color=col, linewidth=3.8 if is_sig else 3.0, alpha=0.92 if is_sig else 0.8, zorder=2)
            if is_sig:
                # Find the After point (x=2) explicitly
                after_idx = None
                for idx, x in enumerate(x_positions):
                    if x == 2:  # After position
                        after_idx = idx
                        break
                if after_idx is not None:
                    ax.scatter(
                        [2.06],  # Slightly to the right of After position
                        [y_values[after_idx]],
                        marker='*',
                        s=360,
                        facecolors='#FFD700',
                        edgecolors='black',
                        linewidth=1.05,
                        zorder=6,
                    )

    # Plot global and per-cluster mean lines
    mean_lines_legend: List[Line2D] = []
    
    # Collect all mean y-positions for smart p-value placement
    mean_y_positions: List[Tuple[str, float, Dict[str, Any]]] = []
    
    # Global mean line
    if not np.isnan(global_stats['mean_before']) and not np.isnan(global_stats['mean_after']):
        mean_x = [1, 2]
        mean_y = [global_stats['mean_before'], global_stats['mean_after']]
        line = ax.plot(mean_x, mean_y, color='black', linewidth=4.5, alpha=0.95, linestyle='--', 
                       marker='D', markersize=10, markerfacecolor='white', markeredgecolor='black', 
                       markeredgewidth=2.5, zorder=7, label=global_stats['label'])[0]
        mean_lines_legend.append(line)
        
        mid_y = (global_stats['mean_before'] + global_stats['mean_after']) / 2
        mean_y_positions.append(('global', mid_y, global_stats))
    
    # Per-cluster mean lines
    cluster_colors_for_means = {
        'A': '#ff3300',  # Bright red-orange for cluster A
        'B': '#0044ff',  # Deep blue for cluster B
    }
    cluster_markers = {'A': 's', 'B': '^'}  # Square for A, triangle for B
    
    for cid in sorted(cluster_stats.keys()):
        stats = cluster_stats[cid]
        if not np.isnan(stats['mean_before']) and not np.isnan(stats['mean_after']):
            mean_x = [1, 2]
            mean_y = [stats['mean_before'], stats['mean_after']]
            color = cluster_colors_for_means.get(cid, '#666666')
            marker = cluster_markers.get(cid, 'o')
            line = ax.plot(mean_x, mean_y, color=color, linewidth=4.0, alpha=0.90, linestyle='-.',
                          marker=marker, markersize=9, markerfacecolor=color, markeredgecolor='black',
                          markeredgewidth=2.0, zorder=7, label=stats['label'])[0]
            mean_lines_legend.append(line)
            
            mid_y = (stats['mean_before'] + stats['mean_after']) / 2
            mean_y_positions.append((cid, mid_y, stats))
    
    # Annotate p-values with smart spacing to avoid overlap
    if mean_y_positions:
        # Sort by y position
        mean_y_positions.sort(key=lambda x: x[1])
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        min_spacing = 0.08 * y_range  # Minimum vertical spacing between labels
        
        adjusted_positions = []
        for i, (label, y_pos, stats) in enumerate(mean_y_positions):
            # Start with the actual position
            adjusted_y = y_pos
            
            # Check overlap with previous labels and adjust
            for prev_y in adjusted_positions:
                if abs(adjusted_y - prev_y) < min_spacing:
                    adjusted_y = prev_y + min_spacing
            
            adjusted_positions.append(adjusted_y)
            
            if not np.isnan(stats['p_value']):
                p_text = f"p={stats['p_value']:.3f}"
                if stats['p_value'] < 0.05:
                    p_text += "*"
                
                if label == 'global':
                    ax.text(1.5, adjusted_y, p_text, fontsize=9, ha='center', va='center', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.5),
                           fontweight='bold', zorder=8)
                else:
                    color = cluster_colors_for_means.get(label, '#666666')
                    ax.text(1.5, adjusted_y, p_text, fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.25', facecolor=color, edgecolor='black', 
                                    linewidth=1.0, alpha=0.7),
                           color='white', fontweight='bold', zorder=8)

    # Axes
    ax.set_xticks([1, 2])
    ax.set_xticklabels([before_label, after_label], fontsize=12)
    channel = plot_data['Channel'].iloc[0] if 'Channel' in plot_data.columns else ''
    stage = plot_data['stage'].iloc[0] if 'stage' in plot_data.columns else 'ALL'
    try:
        logging.info(f"[SPAG_PLOT_AX] channel={channel}, stage={stage}")
    except Exception:
        pass
    ax.set_xlabel('Group', fontsize=14, fontweight='bold')
    if str(metric).upper() == 'AUC':
        ax.set_ylabel(f"Relative AUC (0–1) — {band}", fontsize=14, fontweight='bold')
    else:
        ax.set_ylabel(f"{metric} (a.u.) — {band}", fontsize=14, fontweight='bold')
    ax.set_title(f"Channel {channel} — {band} — Stage {stage}{(' — ' + title_suffix) if title_suffix else ''}\n*: consensus (≥2/3)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.7, axis='y')
    ax.set_facecolor('#FAFAFA')

    # Legend on the right side with better grouping
    try:
        if len(subjects) <= 20:
            if normalized_clusters:
                handles: List[Line2D] = []
                # First add mean lines if present
                if mean_lines_legend:
                    handles.append(Line2D([0], [0], color='none', label='── Means ──'))
                    for mean_line in mean_lines_legend:
                        handles.append(mean_line)
                    handles.append(Line2D([0], [0], color='none', label=''))  # Separator
                
                # Then add clusters and subjects
                handles.append(Line2D([0], [0], color='none', label='── Subjects ──'))
                for cid in cluster_order:
                    members = cluster_subjects.get(cid, [])
                    if not members:
                        continue
                    representative = cluster_rep_colors.get(cid, subject_colors.get(members[0], (0.4, 0.4, 0.4)))
                    cluster_label = cluster_label_map.get(cid, f"Cluster {cid}")
                    handles.append(
                        Line2D(
                            [0], [0],
                            marker='o',
                            linestyle='',
                            markerfacecolor=representative,
                            markeredgecolor='black',
                            label=cluster_label,
                            markersize=6,
                        )
                    )
                    for subj in members:
                        color = subject_colors.get(subj, (0.3, 0.3, 0.3))
                        handles.append(
                            Line2D(
                                [0], [0],
                                marker='o',
                                linestyle='',
                                markerfacecolor=color,
                                markeredgecolor='black',
                                label=f"   {subj}",
                                markersize=4.5,
                            )
                        )
                leg = ax.legend(handles=handles, title='Legend', loc='center left', bbox_to_anchor=(1.02, 0.5), 
                               frameon=True, fontsize=8, borderaxespad=0)
            else:
                handles = []
                # First add mean lines if present
                if mean_lines_legend:
                    handles.append(Line2D([0], [0], color='none', label='── Means ──'))
                    for mean_line in mean_lines_legend:
                        handles.append(mean_line)
                    handles.append(Line2D([0], [0], color='none', label=''))  # Separator
                    handles.append(Line2D([0], [0], color='none', label='── Subjects ──'))
                
                # Then add subjects
                for s in subjects:
                    handles.append(
                        Line2D(
                            [0], [0],
                            marker='o',
                            linestyle='',
                            markerfacecolor=subject_colors[s],
                            markeredgecolor='black',
                            label=str(s),
                            markersize=4.5,
                        )
                    )
                leg = ax.legend(
                    handles=handles,
                    title='Legend',
                    loc='center left',
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=True,
                    fontsize=8,
                    borderaxespad=0,
                )
            if leg and leg.get_frame():
                leg.get_frame().set_alpha(0.95)
                leg.get_frame().set_facecolor('white')
                leg.get_frame().set_edgecolor('black')
                leg.get_frame().set_linewidth(1.0)
    except Exception:
        pass

    # Tableau latéral
    ax_table.axis('off')
    headers, rows, style_meta = _build_table_rows(plot_data, metric=metric, band=band, stats_df=stats_df, subject_colors=subject_colors)
    table_data = [headers] + rows
    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.11, 0.09, 0.09, 0.11, 0.11, 0.09, 0.11, 0.14, 0.07, 0.09, 0.09, 0.09, 0.05, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1, 1.45)
    # Header style
    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_facecolor('#455A64')
        cell.set_text_props(weight='bold', color='white')
    # Décisions colorées and subject names colored by cluster
    for i, meta in enumerate(style_meta, start=1):
        # Color decision cells
        for col_idx, key in [(9, 'perm'), (10, 'boot'), (11, 'z'), (13, 'cons')]:
            cell = table[(i, col_idx)]
            cell.set_facecolor(_color_for_decision(meta[key]))
        # Color subject name (column 0) based on cluster - use exact line color
        if meta.get('subject_color'):
            subject_cell = table[(i, 0)]
            rgb = meta['subject_color']
            # Use the exact line color as background for perfect match
            subject_cell.set_facecolor(rgb)
            # White text for maximum contrast on saturated colors
            subject_cell.set_text_props(weight='bold', color='white')

    ax_table.set_title('Robust intra-subject metrics — codes: UP/DN/=, * = significant', fontsize=11, fontweight='bold', pad=14)

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

    # Return filepath and cluster stats for CSV summary
    return filepath, {'global': global_stats, 'clusters': cluster_stats}


# =====================
# API de génération
# =====================

def _compute_subject_decisions_for_combo(
    plot_data: pd.DataFrame,
    metric: str,
    band: str,
    stats_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, str], float]:
    """
    Calcule la décision par sujet (augmentation/diminution/stagnation) pour une combinaison donnée
    et retourne également une p-value médiane intra-sujet (permutation test) pour le résumé global.
    """
    # Préparer base sujet
    data = plot_data.copy()
    if 'Subject_Base' not in data.columns:
        data['Subject_Base'] = data['Subject'].astype(str).str.extract(r'(S\d+)')[0].fillna(data['Subject'].astype(str))

    # Source pour stats per-epoch si fournie
    stats_src: Optional[pd.DataFrame] = None
    if stats_df is not None and len(stats_df) > 0:
        stats_src = stats_df.copy()
        stats_src['Subject_Base'] = stats_src['Subject'].astype(str).str.extract(r'(S\d+)')[0].fillna(stats_src['Subject'].astype(str))

    decisions: Dict[str, str] = {}
    pvals: List[float] = []
    subjects_unique = data['Subject_Base'].unique()

    # Canal et stade si disponibles (pour filtrage dans stats_src)
    channel = data['Channel'].iloc[0] if 'Channel' in data.columns else None
    stage = data['stage'].iloc[0] if 'stage' in data.columns else None

    for subject in subjects_unique:
        if stats_src is not None:
            filt = (stats_src['Subject_Base'] == subject) & (stats_src['Band'] == band)
            if channel is not None and 'Channel' in stats_src.columns:
                filt = filt & (stats_src['Channel'] == channel)
            if stage is not None and 'stage' in stats_src.columns and 'stage' in data.columns:
                filt = filt & (stats_src['stage'] == stage)
            sub_df = stats_src[filt]
        else:
            sub_df = data[data['Subject_Base'] == subject]

        x_b = sub_df[sub_df['Group'] == 'AVANT'][metric].dropna().to_numpy()
        x_a = sub_df[sub_df['Group'] == 'APRÈS'][metric].dropna().to_numpy()

        perm = permutation_test_median_diff(x_b, x_a)
        boot = bootstrap_ci_median_diff(x_b, x_a)
        rz = robust_z_intrasubject(x_b, x_a)

        decs = [perm['decision'], boot['decision'], rz['decision']]
        num_up = sum(d == 'augmentation' for d in decs)
        num_down = sum(d == 'diminution' for d in decs)
        if num_up >= 2:
            decisions[subject] = 'augmentation'
        elif num_down >= 2:
            decisions[subject] = 'diminution'
        else:
            decisions[subject] = 'stagnation'

        pv = perm.get('p_value', np.nan)
        if not np.isnan(pv):
            pvals.append(float(pv))

    p_med = float(np.median(pvals)) if pvals else float('nan')
    return decisions, p_med


def _export_summary_csv(
    subjects_by_col_and_decision: Dict[str, Dict[str, List[str]]],
    pvals_by_col: Dict[str, List[float]],
    output_dir: str,
    filename: str = "Resume_Comparaison_Bandes_Stades.csv",
    clusters: Optional[Dict[str, str]] = None,
    cluster_names: Optional[Dict[str, str]] = None,
    global_mean_pvals: Optional[Dict[str, List[float]]] = None,
    cluster_pvals: Optional[Dict[str, Dict[str, List[float]]]] = None,
    global_means: Optional[Dict[str, Tuple[float, float, float]]] = None,
    cluster_means: Optional[Dict[str, Dict[str, Tuple[float, float, float]]]] = None,
) -> Optional[str]:
    """
    Exporte un CSV récapitulatif avec:
      - Colonne 0: Indicateur_Global
      - Colonnes suivantes: canal_bande_stade (ex: F3_delta_N3)
      - Ligne 1: tendance majoritaire + p-value Wilcoxon de la moyenne globale
      - Lignes suivantes: augmentation / diminution / stagnation avec la liste des sujets
      - Lignes additionnelles: moyennes before/after et pentes
      - Si clusters fournis: lignes supplémentaires pour tendances par cluster avec p-values et moyennes
    """
    try:
        # Ordonner colonnes par nom
        cols = sorted(subjects_by_col_and_decision.keys())
        header = ["Indicateur_Global"] + cols

        def _trend_and_sig(col: str, subject_list: Optional[List[str]] = None, cluster_id: Optional[str] = None) -> str:
            """Calcule la tendance pour un ensemble de sujets donné."""
            buckets = subjects_by_col_and_decision.get(col, {})
            
            # Si liste de sujets fournie, filtrer pour ce sous-ensemble
            if subject_list is not None:
                subject_set = set(subject_list)
                filtered_buckets = {
                    k: [s for s in v if s in subject_set]
                    for k, v in buckets.items()
                }
                counts = {k: len(set(v)) for k, v in filtered_buckets.items()}
            else:
                counts = {k: len(set(v)) for k, v in buckets.items()}
            
            # Assurer toutes les clés
            for k in ['augmentation', 'diminution', 'stagnation']:
                counts.setdefault(k, 0)
            
            # Tendance majoritaire
            trend = max(counts.keys(), key=lambda k: counts[k]) if counts else 'stagnation'
            
            # P-value
            if subject_list is None:
                # Global: utiliser p-value Wilcoxon de la moyenne globale
                if global_mean_pvals:
                    p_list = global_mean_pvals.get(col, [])
                    if p_list:
                        p_med = float(np.median(p_list))
                        if not np.isnan(p_med):
                            return f"{trend} (p={p_med:.3f}{'*' if p_med < 0.05 else ''})"
                return trend
            elif cluster_id and cluster_pvals:
                # Cluster: utiliser p-value Wilcoxon de la moyenne du cluster
                cluster_p_list = cluster_pvals.get(col, {}).get(cluster_id, [])
                if cluster_p_list:
                    p_med = float(np.median(cluster_p_list))
                    if not np.isnan(p_med):
                        return f"{trend} (p={p_med:.3f}{'*' if p_med < 0.05 else ''})"
                return trend
            else:
                # Pour les clusters sans p-values, juste la tendance
                return trend

        indicator_row = ["Tendance globale"] + [_trend_and_sig(c) for c in cols]
        row_up = ["augmentation"] + [", ".join(sorted(set(subjects_by_col_and_decision.get(c, {}).get('augmentation', [])))) for c in cols]
        row_down = ["diminution"] + [", ".join(sorted(set(subjects_by_col_and_decision.get(c, {}).get('diminution', [])))) for c in cols]
        row_eq = ["stagnation"] + [", ".join(sorted(set(subjects_by_col_and_decision.get(c, {}).get('stagnation', [])))) for c in cols]

        table = [header, indicator_row, row_up, row_down, row_eq]
        
        # Ajouter ligne vide + moyennes globales
        if global_means:
            table.append([""] * len(header))
            # Mean Before
            row_mean_before = ["Mean Before (global)"] + [
                f"{global_means[c][0]:.4f}" if c in global_means else "" for c in cols
            ]
            table.append(row_mean_before)
            # Mean After
            row_mean_after = ["Mean After (global)"] + [
                f"{global_means[c][1]:.4f}" if c in global_means else "" for c in cols
            ]
            table.append(row_mean_after)
            # Slope (After - Before)
            row_slope = ["Slope (Δ global)"] + [
                f"{global_means[c][2]:.4f}" if c in global_means else "" for c in cols
            ]
            table.append(row_slope)

        # Ajouter les tendances par cluster si disponibles
        if clusters and len(clusters) > 0:
            # Normaliser clusters
            normalized_clusters: Dict[str, str] = {}
            for key, cid in clusters.items():
                if key is None or cid is None:
                    continue
                k = str(key).strip().upper()
                if not k:
                    continue
                normalized_clusters[k] = str(cid).strip().upper() or 'A'
            
            # Grouper sujets par cluster
            cluster_subjects: Dict[str, List[str]] = {}
            for subject, cid in normalized_clusters.items():
                cluster_subjects.setdefault(cid, []).append(subject)
            
            # Ajouter ligne vide de séparation
            table.append([""] * len(header))
            
            # Pour chaque cluster, ajouter une ligne de tendance avec p-value et moyennes
            for cid in sorted(cluster_subjects.keys()):
                subjects_in_cluster = cluster_subjects[cid]
                cluster_label = cluster_names.get(cid, f"Cluster {cid}") if cluster_names else f"Cluster {cid}"
                
                # Tendance + p-value
                cluster_trend_row = [f"Tendance {cluster_label}"] + [
                    _trend_and_sig(c, subjects_in_cluster, cluster_id=cid) for c in cols
                ]
                table.append(cluster_trend_row)
                
                # Moyennes du cluster si disponibles
                if cluster_means and any(cid in cluster_means.get(c, {}) for c in cols):
                    # Mean Before
                    row_mean_before = [f"Mean Before ({cluster_label})"] + [
                        f"{cluster_means[c][cid][0]:.4f}" if c in cluster_means and cid in cluster_means[c] else "" for c in cols
                    ]
                    table.append(row_mean_before)
                    # Mean After
                    row_mean_after = [f"Mean After ({cluster_label})"] + [
                        f"{cluster_means[c][cid][1]:.4f}" if c in cluster_means and cid in cluster_means[c] else "" for c in cols
                    ]
                    table.append(row_mean_after)
                    # Slope
                    row_slope = [f"Slope (Δ {cluster_label})"] + [
                        f"{cluster_means[c][cid][2]:.4f}" if c in cluster_means and cid in cluster_means[c] else "" for c in cols
                    ]
                    table.append(row_slope)

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        # Écriture CSV simple
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            import csv
            writer = csv.writer(f)
            for row in table:
                writer.writerow(row)
        try:
            logging.info(f"[SPAG_SUMMARY_CSV] path={out_path}")
        except Exception:
            pass
        return out_path
    except Exception:
        return None
def _filter_metrics(
    metrics: pd.DataFrame,
    selected_bands: Optional[Iterable[str]] = None,
    selected_stages: Optional[Iterable[str]] = None,
    selected_channels: Optional[Iterable[str]] = None,
    selected_subjects: Optional[Iterable[str]] = None,
    selected_band_stage_map: Optional[Dict[str, Iterable[str]]] = None,
    selected_combinations: Optional[Iterable[Tuple[str, str, str]]] = None,
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
    # Filtrage explicite par combinaisons (Channel, stage, Band)
    if selected_combinations:
        allowed_combo = set((str(ch), str(st), str(b)) for ch, st, b in selected_combinations)
        if 'stage' in df.columns:
            df = df[df.apply(lambda r: (str(r['Channel']), str(r['stage']), str(r['Band'])) in allowed_combo, axis=1)]
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
    selected_combinations: Optional[Iterable[Tuple[str, str, str]]] = None,
    bands_mapping: Optional[Dict[str, Tuple[float, float]]] = None,
    total_range: Tuple[float, float] = (0.5, 100.0),
    metric: str = 'AUC',
    clusters: Optional[Dict[str, str]] = None,
    cluster_names: Optional[Dict[str, str]] = None,
    before_label: str = "Before",
    after_label: str = "After",
) -> List[str]:
    # Calcul des métriques à partir des FFT
    metrics = compute_band_metrics_from_fft(df_fft, bands=bands_mapping or DEFAULT_BANDS, total_range=total_range)
    # Filtrage
    metrics = _filter_metrics(metrics, selected_bands, selected_stages, selected_channels, selected_subjects, None, selected_combinations)
    if len(metrics) == 0:
        return []
    # Préparer données pour graphes
    agg = _aggregate_for_plot(metrics, metric=metric)
    outputs: List[str] = []
    # Accumulateur pour résumé global
    subjects_acc: Dict[str, Dict[str, List[str]]]= {}
    pvals_acc: Dict[str, List[float]] = {}
    global_mean_pvals_acc: Dict[str, List[float]] = {}  # {col_key: [global_p_values]}
    global_means_acc: Dict[str, Tuple[float, float, float]] = {}  # {col_key: (mean_before, mean_after, slope)}
    cluster_pvals_acc: Dict[str, Dict[str, List[float]]] = {}  # {col_key: {cluster_id: [p_values]}}
    cluster_means_acc: Dict[str, Dict[str, Tuple[float, float, float]]] = {}  # {col_key: {cluster_id: (before, after, slope)}}
    for ch, st, b in _make_combinations(agg):
        sub = agg[(agg['Channel'] == ch) & (agg['Band'] == b) & (agg['stage'] == st)]
        if len(sub) == 0:
            continue
        # Source per-epoch pour stats
        stats_src = metrics[(metrics['Channel'] == ch) & (metrics['Band'] == b) & (metrics['stage'] == st)]

        # Décisions par sujet pour cette combo
        decisions_map, p_med = _compute_subject_decisions_for_combo(sub, metric=metric, band=b, stats_df=stats_src)
        col_key = f"{str(ch)}_{str(b).lower()}_{str(st)}"  # Inclure le canal
        bucket = subjects_acc.setdefault(col_key, {"augmentation": [], "diminution": [], "stagnation": []})
        for subj, dec in decisions_map.items():
            bucket.setdefault(dec, []).append(str(subj))
        pvals_acc.setdefault(col_key, []).append(p_med)

        out, mean_stats = create_spaghetti_plot(
            sub,
            metric=metric,
            band=b,
            output_dir=os.path.join(output_dir, f"{ch}"),
            title_suffix="",
            full_metrics=stats_src,
            clusters=clusters,
            cluster_names=cluster_names,
            before_label=before_label,
            after_label=after_label,
        )
        if out:
            outputs.append(out)
        
        # Collecter les p-values et moyennes (globale et par cluster)
        if mean_stats:
            # Moyennes et p-value globale
            if 'global' in mean_stats:
                g_stats = mean_stats['global']
                if not np.isnan(g_stats.get('p_value', np.nan)):
                    global_mean_pvals_acc.setdefault(col_key, []).append(g_stats['p_value'])
                if not np.isnan(g_stats.get('mean_before', np.nan)) and not np.isnan(g_stats.get('mean_after', np.nan)):
                    mean_before = g_stats['mean_before']
                    mean_after = g_stats['mean_after']
                    slope = mean_after - mean_before
                    global_means_acc[col_key] = (mean_before, mean_after, slope)
            
            # Moyennes et p-values par cluster
            if 'clusters' in mean_stats:
                cluster_pvals_acc.setdefault(col_key, {})
                cluster_means_acc.setdefault(col_key, {})
                for cid, stats in mean_stats['clusters'].items():
                    if not np.isnan(stats.get('p_value', np.nan)):
                        cluster_pvals_acc[col_key].setdefault(cid, []).append(stats['p_value'])
                    if not np.isnan(stats.get('mean_before', np.nan)) and not np.isnan(stats.get('mean_after', np.nan)):
                        mean_before = stats['mean_before']
                        mean_after = stats['mean_after']
                        slope = mean_after - mean_before
                        cluster_means_acc[col_key][cid] = (mean_before, mean_after, slope)

    # Exporter le résumé global dans le dossier principal
    _export_summary_csv(subjects_acc, pvals_acc, output_dir=output_dir, clusters=clusters, cluster_names=cluster_names, 
                       global_mean_pvals=global_mean_pvals_acc, cluster_pvals=cluster_pvals_acc,
                       global_means=global_means_acc, cluster_means=cluster_means_acc)
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
    clusters: Optional[Dict[str, str]] = None,
    cluster_names: Optional[Dict[str, str]] = None,
    before_label: str = "Before",
    after_label: str = "After",
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
        clusters=clusters,
        cluster_names=cluster_names,
        before_label=before_label,
        after_label=after_label,
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
    selected_combinations: Optional[Iterable[Tuple[str, str, str]]] = None,
    bands_mapping: Optional[Dict[str, Tuple[float, float]]] = None,
    total_range: Tuple[float, float] = (0.5, 45.0),
    epoch_len: float = 30.0,
    metric: str = 'AUC',
    rng_seed: int = RNG_SEED,
    n_perm: int = N_PERM,
    n_boot: int = N_BOOT,
    edf_to_excel_map: Optional[Dict[str, str]] = None,
    clusters: Optional[Dict[str, str]] = None,
    cluster_names: Optional[Dict[str, str]] = None,
    before_label: str = "Before",
    after_label: str = "After",
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
    # Calculer l'union des stades requis depuis la carte bande->stades ou combinaisons explicites
    req_stages = None
    if selected_band_stage_map:
        req_set = set()
        for stages in selected_band_stage_map.values():
            for st in stages:
                req_set.add(st)
        req_stages = sorted(req_set)
    elif selected_combinations:
        req_stages = sorted({st for (_ch, st, _b) in selected_combinations})

    df_b = _edf_to_metrics(edf_before, 'AVANT', selected_channels, mapping, epoch_len, total_range, edf_to_excel_map, req_stages) if edf_before else pd.DataFrame()
    df_a = _edf_to_metrics(edf_after, 'APRÈS', selected_channels, mapping, epoch_len, total_range, edf_to_excel_map, req_stages) if edf_after else pd.DataFrame()
    metrics = pd.concat([df_b, df_a], ignore_index=True)
    try:
        logging.info(f"[SPAG_DIRS_METRICS] total_rows={len(metrics)}")
    except Exception:
        pass

    # Filtres
    metrics = _filter_metrics(metrics, selected_bands, selected_stages, selected_channels, selected_subjects, selected_band_stage_map, selected_combinations)
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
    subjects_acc: Dict[str, Dict[str, List[str]]] = {}
    pvals_acc: Dict[str, List[float]] = {}
    global_mean_pvals_acc: Dict[str, List[float]] = {}  # {col_key: [global_p_values]}
    global_means_acc: Dict[str, Tuple[float, float, float]] = {}  # {col_key: (mean_before, mean_after, slope)}
    cluster_pvals_acc: Dict[str, Dict[str, List[float]]] = {}  # {col_key: {cluster_id: [p_values]}}
    cluster_means_acc: Dict[str, Dict[str, Tuple[float, float, float]]] = {}  # {col_key: {cluster_id: (before, after, slope)}}
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
        # Décisions par sujet pour ce combo
        decisions_map, p_med = _compute_subject_decisions_for_combo(sub, metric=metric, band=b, stats_df=stats_src)
        col_key = f"{str(ch)}_{str(b).lower()}_{str(st)}"  # Inclure le canal
        bucket = subjects_acc.setdefault(col_key, {"augmentation": [], "diminution": [], "stagnation": []})
        for subj, dec in decisions_map.items():
            bucket.setdefault(dec, []).append(str(subj))
        pvals_acc.setdefault(col_key, []).append(p_med)

        out, mean_stats = create_spaghetti_plot(
            sub,
            metric=metric,
            band=b,
            output_dir=os.path.join(output_dir, f"{ch}"),
            title_suffix="",
            full_metrics=stats_src,
            clusters=clusters,
            cluster_names=cluster_names,
            before_label=before_label,
            after_label=after_label,
        )
        if out:
            outputs.append(out)
        
        # Collecter les p-values et moyennes (globale et par cluster)
        if mean_stats:
            # Moyennes et p-value globale
            if 'global' in mean_stats:
                g_stats = mean_stats['global']
                if not np.isnan(g_stats.get('p_value', np.nan)):
                    global_mean_pvals_acc.setdefault(col_key, []).append(g_stats['p_value'])
                if not np.isnan(g_stats.get('mean_before', np.nan)) and not np.isnan(g_stats.get('mean_after', np.nan)):
                    mean_before = g_stats['mean_before']
                    mean_after = g_stats['mean_after']
                    slope = mean_after - mean_before
                    global_means_acc[col_key] = (mean_before, mean_after, slope)
            
            # Moyennes et p-values par cluster
            if 'clusters' in mean_stats:
                cluster_pvals_acc.setdefault(col_key, {})
                cluster_means_acc.setdefault(col_key, {})
                for cid, stats in mean_stats['clusters'].items():
                    if not np.isnan(stats.get('p_value', np.nan)):
                        cluster_pvals_acc[col_key].setdefault(cid, []).append(stats['p_value'])
                    if not np.isnan(stats.get('mean_before', np.nan)) and not np.isnan(stats.get('mean_after', np.nan)):
                        mean_before = stats['mean_before']
                        mean_after = stats['mean_after']
                        slope = mean_after - mean_before
                        cluster_means_acc[col_key][cid] = (mean_before, mean_after, slope)
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

    print("\n=== GLOBAL SUMMARY (consensus per combination) ===")
    label_map = {'augmentation': 'increase', 'diminution': 'decrease', 'stagnation': 'stable'}
    for k, v in decisions_counter.items():
        print(f"  {label_map.get(k, k)}: {v}")
    # Export CSV summary (columns band_stage)
    _export_summary_csv(subjects_acc, pvals_acc, output_dir=output_dir, clusters=clusters, cluster_names=cluster_names, 
                       global_mean_pvals=global_mean_pvals_acc, cluster_pvals=cluster_pvals_acc,
                       global_means=global_means_acc, cluster_means=cluster_means_acc)
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
    selected_combinations: Optional[Iterable[Tuple[str, str, str]]] = None,
    bands_mapping: Optional[Dict[str, Tuple[float, float]]] = None,
    total_range: Tuple[float, float] = (0.5, 45.0),
    epoch_len: float = 30.0,
    metric: str = 'AUC',
    rng_seed: int = RNG_SEED,
    n_perm: int = N_PERM,
    n_boot: int = N_BOOT,
    edf_to_excel_map: Optional[Dict[str, str]] = None,
    clusters: Optional[Dict[str, str]] = None,
    cluster_names: Optional[Dict[str, str]] = None,
    before_label: str = "Before",
    after_label: str = "After",
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
    elif selected_combinations:
        req_stages = sorted({st for (_ch, st, _b) in selected_combinations})

    df_b = _edf_to_metrics(before_files, 'AVANT', selected_channels, mapping, epoch_len, total_range, edf_to_excel_map, req_stages) if before_files else pd.DataFrame()
    df_a = _edf_to_metrics(after_files, 'APRÈS', selected_channels, mapping, epoch_len, total_range, edf_to_excel_map, req_stages) if after_files else pd.DataFrame()
    metrics = pd.concat([df_b, df_a], ignore_index=True)
    try:
        logging.info(f"[SPAG_LIST_METRICS] total_rows={len(metrics)}")
    except Exception:
        pass
    metrics = _filter_metrics(metrics, selected_bands, selected_stages, selected_channels, selected_subjects, selected_band_stage_map, selected_combinations)
    if len(metrics) == 0:
        return []
    agg = _aggregate_for_plot(metrics, metric=metric)
    try:
        logging.info(f"[SPAG_LIST_AGG] combos={len(_make_combinations(agg))}")
    except Exception:
        pass
    outputs: List[str] = []
    subjects_acc: Dict[str, Dict[str, List[str]]] = {}
    pvals_acc: Dict[str, List[float]] = {}
    global_mean_pvals_acc: Dict[str, List[float]] = {}  # {col_key: [global_p_values]}
    global_means_acc: Dict[str, Tuple[float, float, float]] = {}  # {col_key: (mean_before, mean_after, slope)}
    cluster_pvals_acc: Dict[str, Dict[str, List[float]]] = {}  # {col_key: {cluster_id: [p_values]}}
    cluster_means_acc: Dict[str, Dict[str, Tuple[float, float, float]]] = {}  # {col_key: {cluster_id: (before, after, slope)}}
    for ch, st, b in _make_combinations(agg):
        sub = agg[(agg['Channel'] == ch) & (agg['Band'] == b) & (agg['stage'] == st)]
        try:
            logging.info(f"[SPAG_LIST_COMBO] ch={ch}, stage={st}, band={b}, n={len(sub)}")
        except Exception:
            pass
        if len(sub) == 0:
            continue
        stats_src = metrics[(metrics['Channel'] == ch) & (metrics['Band'] == b) & (metrics['stage'] == st)]
        # Décisions par sujet pour cette combo
        decisions_map, p_med = _compute_subject_decisions_for_combo(sub, metric=metric, band=b, stats_df=stats_src)
        col_key = f"{str(ch)}_{str(b).lower()}_{str(st)}"  # Inclure le canal
        bucket = subjects_acc.setdefault(col_key, {"augmentation": [], "diminution": [], "stagnation": []})
        for subj, dec in decisions_map.items():
            bucket.setdefault(dec, []).append(str(subj))
        pvals_acc.setdefault(col_key, []).append(p_med)

        out, mean_stats = create_spaghetti_plot(
            sub,
            metric=metric,
            band=b,
            output_dir=os.path.join(output_dir, f"{ch}"),
            title_suffix="",
            full_metrics=stats_src,
            clusters=clusters,
            cluster_names=cluster_names,
            before_label=before_label,
            after_label=after_label,
        )
        if out:
            outputs.append(out)
        
        # Collecter les p-values et moyennes (globale et par cluster)
        if mean_stats:
            # Moyennes et p-value globale
            if 'global' in mean_stats:
                g_stats = mean_stats['global']
                if not np.isnan(g_stats.get('p_value', np.nan)):
                    global_mean_pvals_acc.setdefault(col_key, []).append(g_stats['p_value'])
                if not np.isnan(g_stats.get('mean_before', np.nan)) and not np.isnan(g_stats.get('mean_after', np.nan)):
                    mean_before = g_stats['mean_before']
                    mean_after = g_stats['mean_after']
                    slope = mean_after - mean_before
                    global_means_acc[col_key] = (mean_before, mean_after, slope)
            
            # Moyennes et p-values par cluster
            if 'clusters' in mean_stats:
                cluster_pvals_acc.setdefault(col_key, {})
                cluster_means_acc.setdefault(col_key, {})
                for cid, stats in mean_stats['clusters'].items():
                    if not np.isnan(stats.get('p_value', np.nan)):
                        cluster_pvals_acc[col_key].setdefault(cid, []).append(stats['p_value'])
                    if not np.isnan(stats.get('mean_before', np.nan)) and not np.isnan(stats.get('mean_after', np.nan)):
                        mean_before = stats['mean_before']
                        mean_after = stats['mean_after']
                        slope = mean_after - mean_before
                        cluster_means_acc[col_key][cid] = (mean_before, mean_after, slope)
    
    # Exporter le résumé global dans le dossier principal
    _export_summary_csv(subjects_acc, pvals_acc, output_dir=output_dir, clusters=clusters, cluster_names=cluster_names, 
                       global_mean_pvals=global_mean_pvals_acc, cluster_pvals=cluster_pvals_acc,
                       global_means=global_means_acc, cluster_means=cluster_means_acc)
    return outputs


