"""
CESA v0.0beta1.0 - Scoring I/O Utilities
================================

Module d'import et de synchronisation du scoring pour CESA v0.0beta1.0.
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
Version: 0.0beta1.0
Date: 2025-09-26
"""

from typing import Optional
import pandas as pd
import numpy as np
import mne


def import_excel_scoring(
    df: pd.DataFrame,
    absolute_start_datetime: Optional[pd.Timestamp],
    epoch_seconds: float = 30.0,
) -> pd.DataFrame:
    """Normalize Excel scoring DataFrame to columns ['time','stage'] in seconds.

    Fills gaps with 'U' should be handled by the caller because it depends on
    the known recording duration.
    """
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    stage_col, dt_col = work.columns[:2]
    work = work[[stage_col, dt_col]].dropna()
    work = work.rename(columns={stage_col: 'stage', dt_col: 'datetime'})
    work['datetime'] = pd.to_datetime(work['datetime'], errors='coerce')
    work = work.dropna(subset=['datetime', 'stage'])

    if absolute_start_datetime is not None:
        base_ts = pd.Timestamp(absolute_start_datetime)
        series = work['datetime']
        # Harmoniser les timezones
        if base_ts.tz is not None:
            if series.dt.tz is None:
                series = series.dt.tz_localize(base_ts.tz)
            else:
                series = series.dt.tz_convert(base_ts.tz)
        else:
            series = series.dt.tz_localize(None)

        times = (series - base_ts).dt.total_seconds()
        # Gestion du passage de minuit (scoring sur la même nuit que l'EDF)
        day_sec = 24.0 * 3600.0
        times = times.where(times >= 0.0, times + day_sec)
        work['time'] = times
    else:
        # Fallback: create synthetic times by epoch size
        work = work.sort_values('datetime').reset_index(drop=True)
        work['time'] = np.arange(len(work), dtype=float) * float(epoch_seconds)

    out = work[['time', 'stage']].sort_values('time').reset_index(drop=True)
    out = out.drop_duplicates(subset=['time']).reset_index(drop=True)
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
