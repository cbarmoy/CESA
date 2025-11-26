"""
CESA v3.0 - PSG Multi-Subplot Viewer
====================================

Modulaire, indépendant de Tkinter, ce module fournit une classe `PSGPlotter`
capable d'afficher une polysomnographie complète sur plusieurs sous-graphes
synchrosés (hypnogramme, EEG, EOG, EMG, ECG/HR, barre d'annotations).

Intégration prévue avec EEGAnalysisStudio via FigureCanvasTkAgg.

Contraintes et objectifs:
- Partage du même axe temps (X) entre tous les sous-graphes
- Navigation/zoom natif Matplotlib; synchronisation via sharex
- Axe X en hh:mm:ss
- Auto-échelle verticale robuste par type de signal
- Marquage d'époques (30s par défaut)
- Coloration par stades (via theme_manager)
- Gestion canaux manquants avec message dans le subplot
- Export PNG/PDF

Auteur: IRBA - CESA
Date: 2025-10-23
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, List, Any

from datetime import datetime
from collections import deque
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from CESA.filters import (
    apply_filter as cesa_apply_filter,
    apply_baseline_correction as cesa_apply_baseline_correction,
    detect_signal_type as cesa_detect_signal_type,
    get_filter_presets as cesa_get_filter_presets,
)
from CESA.theme_manager import theme_manager
from core.telemetry import telemetry


Seconds = float
Signal = Tuple[np.ndarray, float]  # (data, fs)


def _checkpoint_timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _log_checkpoint(message: str) -> None:
    try:
        print(f"{_checkpoint_timestamp()} | {message}", flush=True)
    except Exception:
        pass


def _seconds_to_hhmmss(x: float) -> str:
    x = max(0.0, float(x))
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    s = int(x % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _robust_ylim(data: np.ndarray, default_span: float) -> Tuple[float, float]:
    if data is None or data.size == 0 or not np.any(np.isfinite(data)):
        return (-default_span, default_span)
    clean = np.asarray(data, dtype=float)
    clean = np.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)
    q1, q3 = np.percentile(clean, [5, 95])
    span = max(1e-6, float(q3 - q1))
    center = float((q3 + q1) / 2.0)
    half = max(default_span, span * 0.75)
    return (center - half, center + half)


def _epoch_vlines(ax, start_s: Seconds, end_s: Seconds, epoch_len: Seconds) -> None:
    if epoch_len <= 0:
        return
    t = float(np.ceil(start_s / epoch_len) * epoch_len)
    while t < end_s:
        ax.axvline(t, color="#cccccc", alpha=0.4, linewidth=0.8, zorder=0)
        t += epoch_len


class PSGPlotter:
    """Afficheur PSG multi-subplots, indépendant de l'UI.

    Parameters
    ----------
    signals : dict
        Dictionnaire de signaux: {name: (array, fs)}. Les séries doivent être en µV
        (EEG/EOG/EMG/ECG). Les canaux manquants sont tolérés.
    hypnogram : tuple | None
        (labels: List[str], epoch_length: float). labels[i] ∈ {"W","N1","N2","N3","R","U"}
    scoring_annotations : list | None
        Liste d'événements optionnelle: dicts avec au minimum {"onset": s, "duration": s, "label": str}
    start_time_s : float
        Début de la fenêtre d'affichage en secondes
    duration_s : float
        Durée de la fenêtre en secondes
    filter_params_by_channel : dict | None
        Mapping par canal: {name: {enabled, low, high, amplitude}}. Si None, utilise presets par type.
    global_filter_enabled : bool
        Active/désactive le filtrage appliqué au rendu.
    theme_name : str | None
        Thème à utiliser depuis theme_manager (si None, thème courant).
    """

    def __init__(
        self,
        *,
        signals: Dict[str, Signal],
        hypnogram: Optional[Tuple[List[str], float]] = None,
        scoring_annotations: Optional[List[Dict[str, Any]]] = None,
        start_time_s: Seconds = 0.0,
        duration_s: Seconds = 30.0,
        filter_params_by_channel: Optional[Dict[str, Dict[str, Any]]] = None,
        global_filter_enabled: bool = True,
        theme_name: Optional[str] = None,
        total_duration_s: Optional[float] = None,
    ) -> None:
        self.signals = signals or {}
        self.hypnogram = hypnogram
        self.scoring_annotations = scoring_annotations or []
        self.start_time_s = float(max(0.0, start_time_s))
        self.duration_s = float(max(1.0, duration_s))
        self.filter_params_by_channel = filter_params_by_channel or {}
        self.global_filter_enabled = bool(global_filter_enabled)
        if theme_name is not None:
            theme_manager.set_theme(theme_name)
        self.total_duration_s: Optional[float] = float(total_duration_s) if total_duration_s is not None else None
        # Store last signals for progressive fusion during async preprocessing
        self.last_signals: Dict[str, Signal] = {}
        # Initialize drawing/cache structures BEFORE first redraw
        # Store artists for efficient updates
        self._line_store: Dict[str, Dict[str, Any]] = {"eeg": {}, "eog": {}, "emg": {}, "ecg": {}}
        self._legend_done: Dict[str, bool] = {"eeg": False, "eog": False, "emg": False, "ecg": False}
        # Blitting support (best effort)
        self._use_blit: bool = True
        self._blit_ready: bool = False
        self._axis_backgrounds: Dict[Any, Any] = {}
        # Performance metrics (last frame)
        self._perf_last: Dict[str, float] = {
            "filter_ms": 0.0,
            "baseline_ms": 0.0,
            "draw_ms": 0.0,
            "n_channels": 0.0,
            "n_points": 0.0,
            "decim_ms": 0.0,
        }
        # Track window for background invalidation
        self._last_window: Optional[Tuple[float, float]] = None
        self._invalidate_backgrounds: bool = True
        # Runtime flags propagated from UI
        self.baseline_enabled: bool = True
        self.autoscale_enabled: bool = True
        # Axis limits cache per type when autoscale is disabled
        self._axis_limits: Dict[str, Tuple[float, float]] = {}
        # UI options
        self._line_width: float = 0.9
        self._show_labels: bool = True
        self._show_events: bool = True
        self._grid_on: bool = True
        self._high_contrast: bool = False
        self._nav_cb = None
        # Telemetry helpers
        self._frame_history: deque[float] = deque(maxlen=120)
        self._fps_avg: float = 0.0
        self._current_frame_sample: Optional[Dict[str, object]] = None
        self._signals_preprocessed: bool = False
        # Build figure (triggers first redraw safely)
        self.figure: Figure = self._build_figure()

    # ---------- Public API ----------
    def set_time_window(self, start_time_s: Seconds, duration_s: Seconds) -> None:
        self.start_time_s = float(max(0.0, start_time_s))
        self.duration_s = float(max(1.0, duration_s))
        self._redraw()

    def update_signals(self, signals: Dict[str, Signal]) -> None:
        """Replace current signals with a pre-sliced window and redraw."""
        self._signals_preprocessed = False
        self.signals = signals or {}
        self._redraw()

    def update_preprocessed_signals(self, signals: Dict[str, Signal]) -> None:
        """Update signals already filtered/baselined externally."""
        self._signals_preprocessed = True
        self.signals = signals or {}
        self._redraw()

    def set_theme(self, theme_name: str) -> None:
        theme_manager.set_theme(theme_name)
        self._apply_theme()
        # Invalidate blit backgrounds so new theme paints fully
        self._invalidate_backgrounds = True
        try:
            _log_checkpoint(f"🔍 CHECKPOINT THEME: set_theme -> {theme_name}")
        except Exception:
            pass
        self._redraw(draw_only=False)

    def set_nav_callback(self, callback) -> None:
        """Register a host callback to navigate to a given time (in seconds)."""
        try:
            self._nav_cb = callback
        except Exception:
            self._nav_cb = None

    def set_hypnogram(self, hypnogram: Optional[Tuple[List[str], float]]) -> None:
        """Update hypnogram (labels, epoch_length) and redraw."""
        self.hypnogram = hypnogram
        self._redraw()

    def set_total_duration(self, total_duration_s: float) -> None:
        try:
            self.total_duration_s = float(total_duration_s)
        except Exception:
            self.total_duration_s = None
        self._redraw(draw_only=True)

    def set_global_filter_enabled(self, enabled: bool) -> None:
        try:
            self.global_filter_enabled = bool(enabled)
        except Exception:
            pass
        try:
            _log_checkpoint(f"🔧 CHECKPOINT VIEWER: filter -> {'on' if self.global_filter_enabled else 'off'}")
        except Exception:
            pass
        self._invalidate_backgrounds = True
        self._redraw()

    def set_baseline_enabled(self, enabled: bool) -> None:
        try:
            self.baseline_enabled = bool(enabled)
        except Exception:
            pass
        try:
            _log_checkpoint(f"🔧 CHECKPOINT VIEWER: baseline -> {'on' if self.baseline_enabled else 'off'}")
        except Exception:
            pass
        self._invalidate_backgrounds = True
        self._redraw()

    def set_autoscale_enabled(self, enabled: bool) -> None:
        try:
            self.autoscale_enabled = bool(enabled)
        except Exception:
            pass
        # Invalidate backgrounds because axis limits may change
        self._invalidate_backgrounds = True
        self._redraw()

    def save_png(self, filepath: str, dpi: int = 150) -> None:
        self.figure.savefig(filepath, dpi=dpi, bbox_inches="tight")

    def save_pdf(self, filepath: str) -> None:
        self.figure.savefig(filepath, format="pdf", bbox_inches="tight")

    def export_scoring_csv(self, filepath: str) -> None:
        if self.hypnogram is None:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("time,stage\n")
            return
        labels, epoch_len = self.hypnogram
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("time,stage\n")
            t = 0.0
            for lab in labels:
                f.write(f"{t:.2f},{lab}\n")
                t += float(epoch_len)

    # ---------- Building ----------
    def _build_figure(self) -> Figure:
        # Build figure with independent hypnogram axis (not sharex)
        fig = plt.figure(figsize=(14, 10))
        self.figure = fig
        gs = fig.add_gridspec(6, 1, height_ratios=[1.0, 5.0, 1.2, 1.2, 1.2, 0.6])
        self.ax_hypno = fig.add_subplot(gs[0])
        self.ax_eeg = fig.add_subplot(gs[1])
        self.ax_eog = fig.add_subplot(gs[2], sharex=self.ax_eeg)
        self.ax_emg = fig.add_subplot(gs[3], sharex=self.ax_eeg)
        self.ax_ecg = fig.add_subplot(gs[4], sharex=self.ax_eeg)
        self.ax_events = fig.add_subplot(gs[5], sharex=self.ax_eeg)
        self.axes_all: List[Any] = [self.ax_hypno, self.ax_eeg, self.ax_eog, self.ax_emg, self.ax_ecg, self.ax_events]

        # Remove vertical gaps between subplots
        try:
            fig.subplots_adjust(hspace=0.0, top=0.98, bottom=0.06, left=0.07, right=0.98)
        except Exception:
            pass

        self._apply_theme()
        try:
            _log_checkpoint("🔍 CHECKPOINT FIGURE: figure/axes built, applying first redraw")
        except Exception:
            pass
        # Connect click for hypnogram navigation
        try:
            fig.canvas.mpl_connect('button_press_event', self._on_click)
        except Exception:
            pass
        self._redraw()
        return fig

    def _apply_theme(self) -> None:
        theme_manager.apply_background_to_figure(self.figure)
        # Matplotlib rcParams harmonisés (typo/couleurs axes)
        try:
            import matplotlib as _mpl
            ui = theme_manager.get_current_theme().get_ui_colors()
            _mpl.rcParams.update({
                'font.family': 'DejaVu Sans',
                'axes.facecolor': ui.get('surface', '#ffffff'),
                'axes.edgecolor': ui.get('border', '#dddddd'),
                'axes.labelcolor': ui.get('fg', '#1f2335'),
                'xtick.color': ui.get('fg_muted', '#5b6078'),
                'ytick.color': ui.get('fg_muted', '#5b6078'),
                'grid.color': ui.get('border', '#dddddd'),
                'figure.facecolor': ui.get('bg', '#f7f7fb'),
                'savefig.facecolor': ui.get('bg', '#f7f7fb'),
            })
        except Exception:
            pass
        # Ajuste axes facecolor selon thème
        ui = theme_manager.get_current_theme().get_ui_colors()
        face = ui.get("surface", "#ffffff")
        for ax in getattr(self, "axes_all", []):
            try:
                ax.set_facecolor(face)
                # Harmoniser les couleurs des spines/ticks au thème
                for side in ['top', 'right', 'left', 'bottom']:
                    ax.spines[side].set_color(ui.get('border', '#dddddd'))
                ax.tick_params(colors=ui.get('fg_muted', '#5b6078'))
                ax.yaxis.label.set_color(ui.get('fg', '#1f2335'))
                ax.xaxis.label.set_color(ui.get('fg', '#1f2335'))
            except Exception:
                pass

    # ---------- Rendering ----------
    def _redraw(self, draw_only: bool = False) -> None:
        frame_start = time.perf_counter()
        start_s = self.start_time_s
        end_s = self.start_time_s + self.duration_s
        # Reset perf counters for this frame
        try:
            self._perf_last["filter_ms"] = 0.0
            self._perf_last.setdefault("baseline_ms", 0.0)
            self._perf_last["baseline_ms"] = 0.0
            self._perf_last["draw_ms"] = 0.0
            self._perf_last["n_channels"] = 0
            self._perf_last["n_points"] = 0
            self._perf_last["decim_ms"] = 0.0
        except Exception:
            pass

        frame_sample = telemetry.new_sample(
            {
                "channel": "render",
                "start_s": float(start_s),
                "duration_s": float(self.duration_s),
                "viewport_px": self._estimate_viewport_px(),
                "notes": f"signals={len(self.signals)}",
            }
        )
        self._current_frame_sample = frame_sample if frame_sample else None

        try:
            _log_checkpoint(f"🔍 CHECKPOINT REDRAW: window={start_s:.2f}-{end_s:.2f}s, draw_only={draw_only}, blit_ready={self._blit_ready}, invalidate_bg={self._invalidate_backgrounds}, autoscale={getattr(self, 'autoscale_enabled', True)}, filter={getattr(self, 'global_filter_enabled', True)}, baseline={getattr(self, 'baseline_enabled', True)}")
        except Exception:
            pass

        # Avoid clearing axes; update artists in place for performance
        # Hypnogram and events band are rebuilt by dedicated methods.

        # Hypnogram (independent full overview)
        self._plot_hypnogram(self.ax_hypno, start_s, end_s)

        # Signals by type
        self._plot_signals_of_type(self.ax_eeg, target_type="eeg")
        # Do not overlay stage bars in EEG if we use the dedicated events band for scoring
        self._plot_signals_of_type(self.ax_eog, target_type="eog")
        self._plot_signals_of_type(self.ax_emg, target_type="emg")
        self._plot_signals_of_type(self.ax_ecg, target_type="ecg")

        # Events/scoring band at bottom
        if self._show_events:
            if self.hypnogram is not None:
                self._plot_scoring_band(self.ax_events, start_s, end_s)
            else:
                self._plot_events(self.ax_events, start_s, end_s)
        else:
            try:
                self.ax_events.clear()
                self.ax_events.set_yticks([])
            except Exception:
                pass

        # Common X axis formatting: full duration for hypnogram, window for others
        # Détecter si les limites X ont changé (pour invalider le blitting si nécessaire)
        xlims_changed = False
        for ax in [self.ax_eeg, self.ax_eog, self.ax_emg, self.ax_ecg, self.ax_events]:
            old_xlim = ax.get_xlim()
            ax.set_xlim(start_s, end_s)
            new_xlim = ax.get_xlim()
            if abs(old_xlim[0] - new_xlim[0]) > 0.01 or abs(old_xlim[1] - new_xlim[1]) > 0.01:
                xlims_changed = True
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: _seconds_to_hhmmss(x)))
            _epoch_vlines(ax, start_s, end_s, self._get_epoch_len())
        
        # Invalider le blitting si les limites X ont changé
        if xlims_changed:
            self._invalidate_backgrounds = True

        # Only bottom axis with xlabel to save vertical space
        try:
            for ax in [self.ax_hypno, self.ax_eeg, self.ax_eog, self.ax_emg, self.ax_ecg]:
                ax.set_xlabel("")
                ax.tick_params(axis='x', labelbottom=False)
        except Exception:
            pass
        self.ax_events.set_xlabel("Temps (hh:mm:ss)")

        if not draw_only and self.figure.canvas is not None:
            try:
                _t_draw0 = time.perf_counter()
                canvas = self.figure.canvas
                
                # Détecter si on peut utiliser le blitting
                window_changed = self._last_window is None or abs(self._last_window[0] - start_s) > 0.01 or abs(self._last_window[1] - end_s) > 0.01
                can_use_blit = (self._use_blit and self._blit_ready and not self._invalidate_backgrounds and 
                               not window_changed and not xlims_changed and not self.autoscale_enabled)
                
                if can_use_blit:
                    # Utiliser le blitting pour une mise à jour rapide
                    try:
                        # Activer le mode blitting pour les axes
                        for ax in [self.ax_eeg, self.ax_eog, self.ax_emg, self.ax_ecg]:
                            if ax in self._axis_backgrounds:
                                # Restaurer le background
                                canvas.restore_region(self._axis_backgrounds[ax])
                                # Dessiner seulement les éléments qui ont changé (lignes)
                                renderer = canvas.get_renderer()
                                ax.draw_artist(ax.patch)  # Fond de l'axe
                                ax.draw_artist(ax.xaxis)  # Axe X
                                ax.draw_artist(ax.yaxis)  # Axe Y
                                # Dessiner toutes les lignes
                                for line in ax.lines:
                                    ax.draw_artist(line)
                                # Dessiner les textes (labels de canaux)
                                for txt in ax.texts:
                                    ax.draw_artist(txt)
                                # Dessiner la grille si visible
                                if ax.xaxis._gridOnMajor:
                                    for line in ax.xaxis.get_gridlines():
                                        ax.draw_artist(line)
                                if ax.yaxis._gridOnMajor:
                                    for line in ax.yaxis.get_gridlines():
                                        ax.draw_artist(line)
                                # Blit la région de l'axe
                                canvas.blit(ax.bbox)
                        
                        # Hypnogram et events nécessitent un redraw complet car ils changent souvent
                        # (ils utilisent clear() donc pas de blitting possible)
                        self.ax_hypno.draw(canvas.get_renderer())
                        self.ax_events.draw(canvas.get_renderer())
                        
                        canvas.flush_events()
                        try:
                            self._perf_last["draw_ms"] = (time.perf_counter() - _t_draw0) * 1000.0
                            _log_checkpoint(f"⏱️ DRAW (BLIT): draw_ms={self._perf_last['draw_ms']:.1f}ms")
                        except Exception:
                            pass
                    except Exception as e:
                        # Fallback vers full draw si le blitting échoue
                        try:
                            _log_checkpoint(f"⚠️ BLIT failed: {e}, falling back to full draw")
                        except Exception:
                            pass
                        can_use_blit = False
                        self._blit_ready = False
                        self._invalidate_backgrounds = True
                
                if not can_use_blit:
                    # Full draw nécessaire (changement de fenêtre, limites, ou premier rendu)
                    with telemetry.measure(frame_sample, "render_ms"):
                        canvas.draw()
                    # Refresh backgrounds après draw pour les futurs blits
                    try:
                        for a in [self.ax_eeg, self.ax_eog, self.ax_emg, self.ax_ecg]:
                            self._axis_backgrounds[a] = canvas.copy_from_bbox(a.bbox)
                        self._blit_ready = True
                        self._invalidate_backgrounds = False
                        self._last_window = (start_s, end_s)
                        _log_checkpoint("🔍 CHECKPOINT BLIT: backgrounds refreshed for all axes")
                    except Exception:
                        pass
                    if _t_draw0 is not None:
                        try:
                            self._perf_last["draw_ms"] = (time.perf_counter() - _t_draw0) * 1000.0
                            _log_checkpoint(f"⏱️ DRAW (FULL): draw_ms={self._perf_last['draw_ms']:.1f}ms")
                        except Exception:
                            pass
            except Exception:
                # Fallback: full redraw
                try:
                    with telemetry.measure(frame_sample, "render_ms"):
                        self.figure.canvas.draw()
                except Exception:
                    pass

        frame_duration_s = max(0.0, time.perf_counter() - frame_start)
        if frame_sample:
            try:
                frame_sample.setdefault("filter_ms", float(self._perf_last.get("filter_ms", 0.0)))
                frame_sample.setdefault("decim_ms", float(self._perf_last.get("decim_ms", 0.0)))
                frame_sample.setdefault("render_ms", float(self._perf_last.get("draw_ms", 0.0)))
                frame_sample["viewport_px"] = frame_sample.get("viewport_px") or self._estimate_viewport_px()
                existing = str(frame_sample.get("notes") or "")
                extra = f"n_points={int(self._perf_last.get('n_points', 0.0))}" if self._perf_last.get("n_points") else ""
                if extra:
                    frame_sample["notes"] = ";".join(filter(None, [existing, extra]))
                frame_sample["fps"] = self._update_fps(frame_duration_s)
            except Exception:
                pass
            telemetry.commit(frame_sample)

        self._current_frame_sample = None

    def _get_epoch_len(self) -> float:
        if self.hypnogram is None:
            return 30.0
        return float(self.hypnogram[1]) if len(self.hypnogram) > 1 else 30.0

    def _estimate_viewport_px(self) -> int:
        try:
            canvas = self.figure.canvas
            if canvas is None:
                return 0
            width, _ = canvas.get_width_height()
            return int(width)
        except Exception:
            return 0

    def _update_fps(self, frame_duration_s: float) -> float:
        if frame_duration_s <= 0.0:
            return self._fps_avg
        self._frame_history.append(frame_duration_s)
        avg_duration = sum(self._frame_history) / len(self._frame_history)
        self._fps_avg = 1.0 / avg_duration if avg_duration > 0.0 else 0.0
        return self._fps_avg

    # ---------- Individual Layers ----------
    def _plot_hypnogram(self, ax, start_s: Seconds, end_s: Seconds) -> None:
        # ESA-style hypnogram (step-plot)
        if self.hypnogram is None:
            ax.clear()
            ax.text(0.5, 0.5, "Hypnogramme absent", transform=ax.transAxes, ha="center", va="center")
            ax.set_yticks([])
            _log_checkpoint("🔍 CHECKPOINT HYPNO: none")
            return

        labels, epoch_len = self.hypnogram
        epoch_len = float(epoch_len) if epoch_len else 30.0
        if epoch_len <= 0.0:
            epoch_len = 30.0

        times = np.arange(0.0, len(labels) * epoch_len, epoch_len, dtype=float)
        mapper = {
            'W': 4, 'WAKE': 4, 'ÉVEIL': 4, 'EVEIL': 4, '0': 4,
            'R': 3, 'REM': 3, '5': 3,
            'N1': 2, '1': 2,
            'N2': 1, '2': 1,
            'N3': 0, '3': 0, '4': 0,
            'U': -0.3, 'UNDEFINED': -0.3, 'ARTIFACT': -0.3
        }
        vals = np.array([mapper.get(str(s).upper().strip(), 0) for s in labels], dtype=float)
        if len(times) == 0:
            ax.set_yticks([])
            return
        t_step = np.append(times, times[-1] + epoch_len)
        v_step = np.append(vals, vals[-1])
        ax.clear()
        ui = theme_manager.get_current_theme().get_ui_colors()
        hypno_line = ui.get('fg', '#1f2335')
        ax.plot(t_step, v_step, drawstyle='steps-post', color=hypno_line, linewidth=1.0, alpha=1.0)
        ax.set_ylim(-0.5, 4.5)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(['N3','N2','N1','R','W'])
        # Indication début/fin de scoring (seulement des lignes; pas de texte ni bande)
        try:
            non_u = [i for i, s in enumerate(labels) if str(s).upper().strip() != 'U']
            if len(non_u) > 0:
                s_idx = int(non_u[0])
                e_idx = int(non_u[-1])
                s_t = float(s_idx * epoch_len)
                e_t = float((e_idx + 1) * epoch_len)
                accent = ui.get('accent', '#0d6efd')
                ax.axvline(s_t, color=accent, linestyle='--', linewidth=0.8, alpha=0.9, zorder=3)
                ax.axvline(e_t, color=accent, linestyle='--', linewidth=0.8, alpha=0.9, zorder=3)
        except Exception:
            pass
        # Full-duration X limits for overview and highlight current window
        full_end = self.total_duration_s if self.total_duration_s is not None else (len(labels) * epoch_len)
        ax.set_xlim(0.0, max(epoch_len, full_end))
        ax.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.5)
        ax.set_xlabel("")
        try:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        except Exception:
            pass
        # Highlight current window on hypnogram
        try:
            win_start = float(self.start_time_s)
            win_end = float(self.start_time_s + self.duration_s)
            ax.axvspan(win_start, win_end, color=ui.get('accent', '#0d6efd'), alpha=0.15, zorder=2)
            _log_checkpoint(f"🔍 CHECKPOINT HYPNO: xlim=({0.0:.2f},{full_end:.2f}), highlight={win_start:.2f}-{win_end:.2f}")
        except Exception:
            pass

    def _on_click(self, event) -> None:
        """Click to navigate via hypnogram or events band."""
        try:
            if event.inaxes is None:
                return
            if event.button != 1:
                return
            ax = event.inaxes
            if ax is self.ax_hypno or ax is self.ax_events:
                if event.xdata is None:
                    return
                target_time = float(max(0.0, event.xdata))
                epoch = self._get_epoch_len()
                target_time = float(max(0.0, round(target_time / epoch) * epoch))
                # Host callback if provided
                cb = getattr(self, '_nav_cb', None)
                if callable(cb):
                    cb(target_time)
                else:
                    self.set_time_window(target_time, self.duration_s)
        except Exception:
            pass

    def _plot_signals_of_type(self, ax, *, target_type: str) -> None:
        start_s = self.start_time_s
        end_s = self.start_time_s + self.duration_s

        # Collect matching channels with preferred selection
        # Remove any stale "Aucun canal ..." messages from previous frames
        try:
            for txt in list(ax.texts):
                try:
                    if "Aucun canal" in str(txt.get_text()):
                        txt.remove()
                except Exception:
                    pass
        except Exception:
            pass

        plotted = 0
        channel_names = self._select_channels_for_type(target_type)
        colors = self._channel_colors(channel_names)
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        used_names: List[str] = []
        used_colors: List[Any] = []
        # Decimate to screen resolution (approximate):
        # Estimate ~n_pixels along x using current figure DPI and axis width.
        try:
            bbox = ax.get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
            n_pixels = max(200, int(bbox.width * self.figure.dpi))
        except Exception:
            n_pixels = 800

        target_processing_len = max(300, int(n_pixels * 1.2))

        for idx, ch_name in enumerate(channel_names):
            data_fs = self.signals.get(ch_name, None)
            if data_fs is None:
                continue
            data, fs = data_fs
            x, y = self._slice_and_process(
                ch_name,
                data,
                fs,
                start_s,
                end_s,
                target_len=target_processing_len,
            )
            if x.size == 0:
                continue
            # Decimate if too many points
            if x.size > n_pixels:
                _t_dec = time.perf_counter()
                step = max(1, int(np.floor(x.size / n_pixels)))
                x = x[::step]
                y = y[::step]
                try:
                    self._perf_last["decim_ms"] += (time.perf_counter() - _t_dec) * 1000.0
                except Exception:
                    pass
            xs.append(x)
            ys.append(y)
            used_names.append(ch_name)
            used_colors.append(colors[idx % len(colors)])
            plotted += 1

        try:
            _log_checkpoint(f"🔍 CHECKPOINT PLOT {target_type.upper()}: plotted={plotted}, n_pixels~={n_pixels}")
        except Exception:
            pass

        if plotted == 0 and len(channel_names) == 0:
            ax.text(0.5, 0.5, f"Aucun canal {target_type.upper()} disponible", transform=ax.transAxes,
                    ha="center", va="center")
        else:
            # ESA-style stacking with vertical offsets for readability
            # Determine a reasonable spacing based on robust amplitude
            spans = []
            for y in ys:
                if y.size > 0:
                    q1, q3 = np.percentile(y, [5, 95])
                    spans.append(float(q3 - q1))
            if len(spans) == 0:
                spans = [30.0]
            base = np.median(spans) if len(spans) > 0 else 30.0
            if target_type == "eeg":
                spacing = max(40.0, base * 1.2)
            elif target_type in ("eog", "emg"):
                spacing = max(25.0, base * 1.2)
            else:
                spacing = max(20.0, base * 1.2)

            # Reuse or create lines; update with set_data for speed
            store = self._line_store[target_type]
            existing_names = set(store.keys())
            used_set = set(used_names)
            # Hide lines that are no longer used
            for ch in list(existing_names - used_set):
                try:
                    store[ch].set_data([], [])
                except Exception:
                    pass
            # Update or create lines
            for i, (x, y, name, color) in enumerate(zip(xs, ys, used_names, used_colors)):
                y_off = y + i * spacing
                line = store.get(name)
                if line is None:
                    alpha = 1.0 if getattr(self, '_high_contrast', False) else 0.95
                    lw = float(getattr(self, '_line_width', 0.9))
                    line, = ax.plot(x, y_off, lw=lw, color=color, alpha=alpha)
                    store[name] = line
                else:
                    line.set_data(x, y_off)
                    line.set_color(color)
                    try:
                        line.set_linewidth(float(getattr(self, '_line_width', 0.9)))
                        line.set_alpha(1.0 if getattr(self, '_high_contrast', False) else 0.95)
                    except Exception:
                        pass
                # Channel label: draw once as annotation if not present
                if not hasattr(line, '_label_text'):
                    try:
                        ui = theme_manager.get_current_theme().get_ui_colors()
                        if self._show_labels:
                            txt = ax.text(x[0], i * spacing, name, fontsize=7, va='center', ha='left',
                                          bbox=dict(boxstyle="round,pad=0.2", facecolor=ui.get('surface', '#ffffff'), alpha=0.5),
                                          color=ui.get('fg', '#1f2335'))
                            line._label_text = txt
                        else:
                            line._label_text = None
                    except Exception:
                        pass
                else:
                    try:
                        if line._label_text is not None:
                            line._label_text.set_position((x[0] if x.size else 0.0, i * spacing))
                    except Exception:
                        pass
            # Set y limits: autoscale if enabled, else reuse cached limits
            if self.autoscale_enabled:
                y_min = -spacing * 0.3
                y_max = spacing * max(1.0, (len(ys) - 1) + 0.8)
                ax.set_ylim(y_min, y_max)
                self._axis_limits[target_type] = (y_min, y_max)
            else:
                lim = self._axis_limits.get(target_type, None)
                if lim is None:
                    y_min = -spacing * 0.3
                    y_max = spacing * max(1.0, (len(ys) - 1) + 0.8)
                    lim = (y_min, y_max)
                    self._axis_limits[target_type] = lim
                ax.set_ylim(*lim)
            ax.set_yticks([])
            try:
                cur_ylim = ax.get_ylim()
                _log_checkpoint(f"🔍 CHECKPOINT LIMITS {target_type.upper()}: ylim={cur_ylim}, autoscale={self.autoscale_enabled}")
            except Exception:
                pass
            # Grid
            try:
                ax.grid(bool(getattr(self, '_grid_on', True)), alpha=0.2, linestyle='-', linewidth=0.5)
            except Exception:
                pass
            # Capture backgrounds for blitting (once)
            try:
                if self._use_blit and self.figure.canvas is not None and not self._blit_ready:
                    self.figure.canvas.draw()
                    for a in [self.ax_eeg, self.ax_eog, self.ax_emg, self.ax_ecg, self.ax_events]:
                        self._axis_backgrounds[a] = self.figure.canvas.copy_from_bbox(a.bbox)
                    self._blit_ready = True
            except Exception:
                self._blit_ready = False
            # Update perf counters
            try:
                import numpy as _np
                self._perf_last["n_channels"] = float(len(ys))
                # Sum of points across lines
                self._perf_last["n_points"] = float(sum((_np.asarray(x).size for x in xs)))
            except Exception:
                pass

        frame_sample = self._current_frame_sample
        if frame_sample:
            try:
                detail = f"{target_type}:{plotted}ch"
                existing = str(frame_sample.get("notes") or "")
                if detail not in existing:
                    frame_sample["notes"] = ";".join(filter(None, [existing, detail]))
            except Exception:
                pass

        # For stacked view, we already set y-limits and removed y-ticks

    def _normalize_name(self, name: str) -> str:
        s = (name or "").upper()
        for ch in ("-", "/", " ", "_", ":"):
            s = s.replace(ch, "")
        return s

    def _select_channels_for_type(self, target_type: str) -> List[str]:
        # Available channels of this type
        available = [ch for ch in self.signals.keys() if cesa_detect_signal_type(ch) == target_type]
        if target_type == "eeg":
            # Preferred montage order with fallbacks
            preferred = ["F3M2", "F4M1", "C3M2", "C4M1", "O1M2", "O2M1"]
            norm_map = {self._normalize_name(ch): ch for ch in available}
            picked: List[str] = []
            for p in preferred:
                if p in norm_map:
                    picked.append(norm_map[p])
            # If none picked, use all available in stable order
            return picked if picked else available
        if target_type == "eog":
            preferred = ["E1M2", "E2M1"]
            norm_map = {self._normalize_name(ch): ch for ch in available}
            picked = [norm_map[p] for p in preferred if p in norm_map]
            return picked if picked else available
        if target_type == "emg":
            # Prefer left/right leg if present
            legs = [ch for ch in available if "LEFT LEG" in ch.upper() or "RIGHT LEG" in ch.upper()]
            return legs if legs else available
        return available

    def _channel_colors(self, channel_names: List[str]) -> List[str]:
        """Assign colors per EEG/EOG/EMG pair with dark/light variants and a fixed ECG color.

        Pairs:
          - F3/F4 (blue base)
          - C3/C4 (green base)
          - O1/O2 (purple base)
          - E1/E2 (teal base)
          - Left/Right Leg (amber base)
          - ECG/Heart Rate (red)
        Unknown channels fall back to a qualitative palette.
        """
        import re
        try:
            import matplotlib.colors as mcolors
        except Exception:
            mcolors = None

        base_for_pair: Dict[str, str] = {
            'F': '#3B82F6',      # blue
            'C': '#22C55E',      # green
            'O': '#8B5CF6',      # purple
            'E': '#06B6D4',      # teal
            'LEG': '#F59E0B',    # amber (Left/Right Leg)
        }
        ECG_COLOR = '#EF4444'   # red
        fallback_palette = ['#4a90e2', '#e67e22', '#2ecc71', '#e74c3c', '#9b59b6', '#16a085', '#f1c40f', '#34495e']

        def shade(hex_color: str, factor: float) -> str:
            if not mcolors:
                return hex_color
            r, g, b = mcolors.to_rgb(hex_color)
            r = min(1.0, max(0.0, r * factor))
            g = min(1.0, max(0.0, g * factor))
            b = min(1.0, max(0.0, b * factor))
            return mcolors.to_hex((r, g, b))

        def normalize_full(ch: str) -> str:
            s = (ch or '').upper()
            s = re.sub(r'[\-_/:]', ' ', s)
            return s

        def root_token(full_s: str) -> str:
            tok = full_s.split()[0] if full_s.split() else full_s
            return tok

        def pair_key_and_side(full_s: str, tok: str) -> Tuple[str, str]:
            # EEG pairs
            if tok.startswith('F3'): return ('F', 'L')
            if tok.startswith('F4'): return ('F', 'R')
            if tok.startswith('C3'): return ('C', 'L')
            if tok.startswith('C4'): return ('C', 'R')
            if tok.startswith('O1'): return ('O', 'L')
            if tok.startswith('O2'): return ('O', 'R')
            if tok.startswith('E1'): return ('E', 'L')
            if tok.startswith('E2'): return ('E', 'R')
            # Legs
            if 'LEFT LEG' in full_s: return ('LEG', 'L')
            if 'RIGHT LEG' in full_s: return ('LEG', 'R')
            # ECG / Heart
            if tok.startswith('ECG') or tok.startswith('HEART') or tok == 'HEARTRATE':
                return ('ECG', 'N')
            return ('', '')

        colors: List[str] = []
        fallback_i = 0
        for ch in channel_names:
            full_s = normalize_full(ch)
            tok = root_token(full_s)
            pk, side = pair_key_and_side(full_s, tok)
            if pk == 'ECG':
                colors.append(ECG_COLOR)
                continue
            base = base_for_pair.get(pk, None)
            if base is None:
                colors.append(fallback_palette[fallback_i % len(fallback_palette)])
                fallback_i += 1
                continue
            # Left: darker, Right: lighter
            factor = 0.85 if side == 'L' else 1.15
            colors.append(shade(base, factor))
        return colors

    def _slice_and_process(
        self,
        ch_name: str,
        data: np.ndarray,
        fs: float,
        start_s: Seconds,
        end_s: Seconds,
        *,
        target_len: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if data is None or fs is None:
            return np.array([]), np.array([])
        data = np.asarray(data).astype(float).squeeze()
        fs = float(fs)
        n = data.size
        if n == 0 or fs <= 0.0:
            return np.array([]), np.array([])

        # Always treat provided data as the current window content
        # Build absolute time vector anchored at the requested start time.
        # This avoids transient empty slices when UI duration changes between async updates.
        x = start_s + (np.arange(n, dtype=float) / fs)
        y = data

        if target_len and target_len > 0 and y.size > target_len:
            bin_size = int(np.ceil(y.size / float(target_len)))
            if bin_size > 1:
                usable = y[: (y.size // bin_size) * bin_size]
                down = usable.reshape(-1, bin_size).mean(axis=1).astype(np.float32, copy=False)
                if usable.size < y.size:
                    remainder_mean = float(np.mean(y[usable.size :], dtype=np.float32))
                    down = np.concatenate([down, np.array([remainder_mean], dtype=np.float32)])
                y = down
                fs = fs / bin_size
                x = start_s + (np.arange(y.size, dtype=float) / fs)

        # Baseline and filtering per channel
        params = self.filter_params_by_channel.get(ch_name, None)
        if params is None:
            # Use presets by type
            stype = cesa_detect_signal_type(ch_name)
            presets = cesa_get_filter_presets(stype)
            params = {
                "enabled": True,
                "low": presets.get("low", 0.0),
                "high": presets.get("high", 0.0),
                "amplitude": presets.get("amplitude", 100.0),
            }

        baseline_elapsed = 0.0
        filter_elapsed = 0.0

        if not self._signals_preprocessed:
            try:
                import time as _time
                _t0 = _time.perf_counter()
                if self.baseline_enabled and cesa_detect_signal_type(ch_name) in ("eeg", "eog", "ecg", "emg"):
                    y = cesa_apply_baseline_correction(y, window_duration=30.0, sfreq=fs)
                baseline_elapsed = (_time.perf_counter() - _t0) * 1000.0
            except Exception:
                baseline_elapsed = 0.0

            try:
                import time as _time
                _t0 = _time.perf_counter()
                if self.global_filter_enabled and params.get("enabled", False):
                    y = cesa_apply_filter(
                        y,
                        sfreq=fs,
                        filter_order=4,
                        low=params.get("low", 0.0),
                        high=params.get("high", 0.0),
                    )
                filter_elapsed = (_time.perf_counter() - _t0) * 1000.0
                try:
                    self._perf_last["filter_ms"] += filter_elapsed
                    self._perf_last["baseline_ms"] += baseline_elapsed
                    if y.size > 0 and (hash(ch_name) % 5 == 0):
                        _log_checkpoint(
                            f"⏱️ PREPROC {ch_name}: len={y.size}, baseline_ms={self._perf_last['baseline_ms']:.1f}, filter_ms={self._perf_last['filter_ms']:.1f}"
                        )
                except Exception:
                    pass
            except Exception:
                pass

            try:
                amp = float(params.get("amplitude", 100.0))
                y = y * (amp / 100.0)
            except Exception:
                pass

            if self._current_frame_sample:
                try:
                    self._current_frame_sample.setdefault("baseline_ms", 0.0)
                    self._current_frame_sample.setdefault("filter_ms", 0.0)
                    self._current_frame_sample["baseline_ms"] += baseline_elapsed
                    self._current_frame_sample["filter_ms"] += filter_elapsed
                except Exception:
                    pass
        else:
            # Preprocessed signals are assumed to have already applied filter/baseline/amplitude
            pass

        return x, y

    def _plot_events(self, ax, start_s: Seconds, end_s: Seconds) -> None:
        # Clear axis to avoid residual texts
        ax.clear()
        ax.set_yticks([])
        if not self.scoring_annotations:
            ax.text(0.5, 0.5, "Aucun événement", transform=ax.transAxes, ha="center", va="center")
            return
        y = 0.5
        height = 0.6
        for ev in self.scoring_annotations:
            try:
                onset = float(ev.get("onset", 0.0))
                duration = float(ev.get("duration", 0.0))
                label = str(ev.get("label", ""))
                if onset >= end_s or (onset + duration) <= start_s:
                    continue
                d_start = max(onset, start_s)
                d_end = min(onset + duration, end_s)
                width = max(0.0, d_end - d_start)
                if width <= 0.0:
                    continue
                ax.barh(y, width, left=d_start, height=height, color="#9467bd", alpha=0.7, edgecolor="black", linewidth=0.3)
                ax.text(d_start + width / 2.0, y, label, ha="center", va="center", fontsize=8)
            except Exception:
                continue

    def _plot_scoring_band(self, ax, start_s: Seconds, end_s: Seconds) -> None:
        # Clear axis to remove any previous 'Aucun événement' and redraw cleanly
        ax.clear()
        ax.set_yticks([])
        if self.hypnogram is None:
            self._plot_events(ax, start_s, end_s)
            return
        labels, epoch_len = self.hypnogram
        try:
            epoch_len = float(epoch_len)
        except Exception:
            epoch_len = 30.0
        if epoch_len <= 0:
            epoch_len = 30.0
        times = np.arange(0.0, len(labels) * epoch_len, epoch_len, dtype=float)
        epoch_start = times
        epoch_end = times + epoch_len
        mask = (epoch_start < end_s) & (epoch_end > start_s)
        stage_colors = theme_manager.get_stage_colors().copy()
        ui = theme_manager.get_current_theme().get_ui_colors()
        stage_colors.setdefault('U', '#8c564b')
        # Short labels for display
        short_map = { 'W': 'E', 'WAKE': 'E', 'ÉVEIL': 'E', 'EVEIL': 'E',
                      'N1': 'N1', 'N2': 'N2', 'N3': 'N3', 'R': 'R', 'REM': 'R', 'U': 'U' }
        y = 0.5
        height = 0.9
        any_drawn = False
        for i in np.where(mask)[0]:
            d_start = float(max(epoch_start[i], start_s))
            d_end = float(min(epoch_end[i], end_s))
            width = max(0.0, d_end - d_start)
            if width <= 0.0:
                continue
            lab = str(labels[i]).upper().strip()
            color = stage_colors.get(lab, stage_colors.get('U', '#8c564b'))
            ax.barh(y, width, left=d_start, height=height, color=color, alpha=0.85, edgecolor=ui.get('border', '#222222'), linewidth=0.2)
            # Add centered letter label
            try:
                center_x = d_start + width * 0.5
                txt = short_map.get(lab, lab)
                ax.text(center_x, y, txt, ha='center', va='center', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=ui.get('surface', '#ffffff'), alpha=0.6), color=ui.get('fg', '#1f2335'))
            except Exception:
                pass
            any_drawn = True
        if not any_drawn:
            ax.text(0.5, 0.5, "Aucun scoring dans la fenêtre", transform=ax.transAxes, ha='center', va='center')

    def _plot_stage_bars_on_axis(self, ax, start_s: Seconds, end_s: Seconds) -> None:
        if self.hypnogram is None:
            return
        labels, epoch_len = self.hypnogram
        epoch_len = float(epoch_len) if epoch_len else 30.0
        if epoch_len <= 0:
            epoch_len = 30.0
        times = np.arange(0.0, len(labels) * epoch_len, epoch_len, dtype=float)
        epoch_start = times
        epoch_end = times + epoch_len
        mask = (epoch_start < end_s) & (epoch_end > start_s)
        if not np.any(mask):
            return
        stage_colors = theme_manager.get_stage_colors().copy()
        ui = theme_manager.get_current_theme().get_ui_colors()
        stage_colors.setdefault('U', '#8c564b')
        y_min, y_max = ax.get_ylim()
        bar_h = (y_max - y_min) * 0.03
        y_base = y_min + bar_h * 0.2
        for i in np.where(mask)[0]:
            d_start = float(max(epoch_start[i], start_s))
            d_end = float(min(epoch_end[i], end_s))
            width = max(0.0, d_end - d_start)
            if width <= 0.0:
                continue
            lab = str(labels[i]).upper().strip()
            color = stage_colors.get(lab, stage_colors.get('U', '#8c564b'))
            ax.barh(y_base, width, left=d_start, height=bar_h, color=color, alpha=0.7, edgecolor=ui.get('border', '#222222'), linewidth=0.2, zorder=0)
