"""Professional filter-configuration dialog for CESA (v4).

Complete redesign inspired by Nocturnal / RemLogic / SleepSign with:

* Dark / light theme toggle
* Left panel   -- channel search, multi-select, group buttons, filter-tag badges
* Centre panel -- collapsible filter cards with sliders, physiological range
                  indicators, drag-handle visual cues, per-channel annotations
* Right panel  -- 3-subplot preview (signal overlay, frequency response, PSD)
                  with SNR indicator and artifact highlight
* Top bar      -- preset dropdown (filtered by type), favorite star toggle,
                  save / delete / import / export, adaptive suggestions, undo/redo
* Bottom bar   -- Apply / Reset / Cancel, status summary, audit export,
                  HTML report export, mini-dashboard toggle
* Keyboard     -- Ctrl+Z undo, Ctrl+Y redo, Ctrl+F focus search
* Audit trail  -- every change recorded with timestamps
"""

from __future__ import annotations

import logging
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from CESA.filter_engine import (
    PHYSIOLOGICAL_RANGES,
    AdaptiveFilterSuggester,
    BaseFilter,
    BandpassFilter,
    ChannelAnnotationStore,
    FavoritePresets,
    FilterAuditLog,
    FilterPipeline,
    FilterPreset,
    FilterSuggestion,
    HighpassFilter,
    LowpassFilter,
    NotchFilter,
    PresetLibrary,
    SmoothingFilter,
    UndoManager,
    filter_from_dict,
    pipeline_from_legacy_params,
)

logger = logging.getLogger(__name__)

_FILTER_TYPES: Dict[str, type] = {
    "Passe-bande": BandpassFilter,
    "Passe-haut": HighpassFilter,
    "Passe-bas": LowpassFilter,
    "Notch (rejet)": NotchFilter,
    "Lissage": SmoothingFilter,
}

_FILTER_TAG_COLORS: Dict[str, str] = {
    "BandpassFilter": "#3B82F6",
    "HighpassFilter": "#8B5CF6",
    "LowpassFilter": "#06B6D4",
    "NotchFilter":    "#F59E0B",
    "SmoothingFilter": "#10B981",
}

_DEFAULT_PRESETS_PATH = Path(__file__).resolve().parent.parent / "config" / "filter_presets.json"

_PREVIEW_DEBOUNCE_MS = 150

_CHANNEL_GROUPS: Dict[str, List[str]] = {
    "Tous":  [],
    "EEG":   ["eeg"],
    "EOG":   ["eog"],
    "EMG":   ["emg"],
    "ECG":   ["ecg"],
}

# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------

_THEMES: Dict[str, Dict[str, str]] = {
    "dark": {
        "bg":        "#1E1E2E",
        "bg_alt":    "#2A2A3C",
        "fg":        "#CDD6F4",
        "fg_dim":    "#6C7086",
        "accent":    "#89B4FA",
        "warn":      "#F9E2AF",
        "error":     "#F38BA8",
        "ok":        "#A6E3A1",
        "card_bg":   "#313244",
        "entry_bg":  "#45475A",
        "border":    "#585B70",
        "plot_bg":   "#1E1E2E",
        "plot_fg":   "#CDD6F4",
        "plot_grid":  "#45475A",
        "plot_raw":  "#6C7086",
        "plot_filt": "#89B4FA",
        "plot_freq": "#A6E3A1",
        "plot_psd":  "#CBA6F7",
    },
    "light": {
        "bg":        "#FFFFFF",
        "bg_alt":    "#F1F5F9",
        "fg":        "#1E293B",
        "fg_dim":    "#94A3B8",
        "accent":    "#3B82F6",
        "warn":      "#D97706",
        "error":     "#DC2626",
        "ok":        "#16A34A",
        "card_bg":   "#F8FAFC",
        "entry_bg":  "#FFFFFF",
        "border":    "#CBD5E1",
        "plot_bg":   "#FFFFFF",
        "plot_fg":   "#1E293B",
        "plot_grid":  "#E2E8F0",
        "plot_raw":  "#CBD5E1",
        "plot_filt": "#3B82F6",
        "plot_freq": "#16A34A",
        "plot_psd":  "#7C3AED",
    },
}


def _short_filter_label(filt: BaseFilter) -> str:
    """Short human-readable tag for a filter."""
    name = type(filt).__name__
    if isinstance(filt, BandpassFilter):
        return f"BP {filt.low_hz}-{filt.high_hz}"
    if isinstance(filt, HighpassFilter):
        return f"HP {filt.cutoff_hz}"
    if isinstance(filt, LowpassFilter):
        return f"LP {filt.cutoff_hz}"
    if isinstance(filt, NotchFilter):
        return f"N {filt.freq_hz}"
    if isinstance(filt, SmoothingFilter):
        return f"Sm {filt.method[:3]}"
    return name[:6]


# ---------------------------------------------------------------------------
# Small re-usable widgets
# ---------------------------------------------------------------------------

class _LabeledScale(ttk.Frame):
    """Horizontal slider with label, numeric entry, and physiological range indicator."""

    def __init__(
        self,
        master: tk.Widget,
        label: str,
        from_: float,
        to: float,
        resolution: float = 0.1,
        variable: Optional[tk.DoubleVar] = None,
        on_change: Optional[Callable] = None,
        width_label: int = 12,
        warn_range: Optional[Tuple[float, float]] = None,
        **kw,
    ):
        super().__init__(master, **kw)
        self._var = variable or tk.DoubleVar(value=from_)
        self._on_change = on_change
        self._resolution = resolution
        self._warn_lo, self._warn_hi = warn_range or (None, None)

        ttk.Label(self, text=label, width=width_label, anchor=tk.W).pack(side=tk.LEFT, padx=(0, 2))

        self._scale = ttk.Scale(
            self, from_=from_, to=to, variable=self._var, orient=tk.HORIZONTAL,
            command=self._on_scale, length=120,
        )
        self._scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self._entry = ttk.Entry(self, width=7, justify=tk.RIGHT)
        self._entry.pack(side=tk.LEFT)
        self._entry.insert(0, self._fmt(self._var.get()))
        self._entry.bind("<Return>", self._on_entry)
        self._entry.bind("<FocusOut>", self._on_entry)

        self._warn_label = ttk.Label(self, text="", width=2)
        self._warn_label.pack(side=tk.LEFT, padx=(1, 0))
        self._update_warn_indicator()

    def _fmt(self, val: float) -> str:
        if self._resolution >= 1:
            return str(int(round(val)))
        return f"{val:.2f}"

    def _on_scale(self, _val: str) -> None:
        v = self._var.get()
        if self._resolution >= 1:
            v = round(v)
            self._var.set(v)
        self._entry.delete(0, tk.END)
        self._entry.insert(0, self._fmt(v))
        self._update_warn_indicator()
        if self._on_change:
            self._on_change()

    def _on_entry(self, _event: Any = None) -> None:
        try:
            v = float(self._entry.get())
            self._var.set(v)
            self._update_warn_indicator()
            if self._on_change:
                self._on_change()
        except ValueError:
            pass

    def _update_warn_indicator(self) -> None:
        if self._warn_lo is None or self._warn_hi is None:
            self._warn_label.configure(text="")
            return
        v = self._var.get()
        if v < self._warn_lo or v > self._warn_hi:
            self._warn_label.configure(text="\u26A0", foreground="#E67E22")
        else:
            self._warn_label.configure(text="\u2713", foreground="#27AE60")

    @property
    def var(self) -> tk.DoubleVar:
        return self._var


# ---------------------------------------------------------------------------
# Filter card
# ---------------------------------------------------------------------------

class _FilterCard(ttk.LabelFrame):
    """Editable, collapsible card for a single filter in the pipeline."""

    def __init__(
        self,
        master: tk.Widget,
        filt: BaseFilter,
        index: int,
        channel_type: str,
        on_change: Callable,
        on_delete: Callable,
        on_move_up: Callable,
        on_move_down: Callable,
    ):
        label = type(filt).__name__.replace("Filter", "")
        super().__init__(master, text=f"  {index + 1}. {label}  ", padding=6)
        self.filt = filt
        self._on_change = on_change
        self._channel_type = channel_type.lower() if channel_type else "generic"
        self._collapsed = False

        hdr = ttk.Frame(self)
        hdr.pack(fill=tk.X, pady=(0, 4))

        self._enabled_var = tk.BooleanVar(value=filt.enabled)
        ttk.Checkbutton(hdr, text="Actif", variable=self._enabled_var,
                        command=self._toggle_enabled).pack(side=tk.LEFT)

        ttk.Button(hdr, text="\u2715", width=3, command=on_delete).pack(side=tk.RIGHT, padx=2)
        ttk.Button(hdr, text="\u25BC", width=3, command=on_move_down).pack(side=tk.RIGHT, padx=2)
        ttk.Button(hdr, text="\u25B2", width=3, command=on_move_up).pack(side=tk.RIGHT, padx=2)

        self._collapse_btn = ttk.Button(hdr, text="\u25B3", width=3, command=self._toggle_collapse)
        self._collapse_btn.pack(side=tk.RIGHT, padx=2)

        color = _FILTER_TAG_COLORS.get(type(filt).__name__, "#888")
        tag = tk.Label(hdr, text=_short_filter_label(filt), bg=color,
                       fg="white", font=("Segoe UI", 7, "bold"), padx=4, pady=1)
        tag.pack(side=tk.RIGHT, padx=4)

        self._body = ttk.Frame(self)
        self._body.pack(fill=tk.X)
        self._build_params()

    def _toggle_enabled(self) -> None:
        self.filt.enabled = self._enabled_var.get()
        self._on_change()

    def _toggle_collapse(self) -> None:
        self._collapsed = not self._collapsed
        if self._collapsed:
            self._body.pack_forget()
            self._collapse_btn.configure(text="\u25BD")
        else:
            self._body.pack(fill=tk.X)
            self._collapse_btn.configure(text="\u25B3")

    def _build_params(self) -> None:
        f = self.filt
        if isinstance(f, BandpassFilter):
            self._add_hz_slider("Bas (Hz)", 0.01, 200, f.low_hz, "low_hz", physio_key="bp_low_hz")
            self._add_hz_slider("Haut (Hz)", 0.1, 500, f.high_hz, "high_hz", physio_key="bp_high_hz")
            self._add_order_spinner("Ordre", f.order, "order")
            self._add_type_combo(f.filter_type, "filter_type")
            self._add_causal_check(f.causal, "causal")
        elif isinstance(f, HighpassFilter):
            self._add_hz_slider("Coupure (Hz)", 0.01, 200, f.cutoff_hz, "cutoff_hz", physio_key="hp_cutoff_hz")
            self._add_order_spinner("Ordre", f.order, "order")
            self._add_type_combo(f.filter_type, "filter_type")
            self._add_causal_check(f.causal, "causal")
        elif isinstance(f, LowpassFilter):
            self._add_hz_slider("Coupure (Hz)", 0.1, 500, f.cutoff_hz, "cutoff_hz", physio_key="lp_cutoff_hz")
            self._add_order_spinner("Ordre", f.order, "order")
            self._add_type_combo(f.filter_type, "filter_type")
            self._add_causal_check(f.causal, "causal")
        elif isinstance(f, NotchFilter):
            self._add_hz_slider("Fréquence (Hz)", 1, 500, f.freq_hz, "freq_hz", physio_key="notch_hz")
            self._add_hz_slider("Facteur Q", 1, 100, f.quality_factor, "quality_factor", res=1)
            self._add_int_spinner("Harmoniques", 1, 5, f.harmonics, "harmonics")
        elif isinstance(f, SmoothingFilter):
            method_row = ttk.Frame(self._body)
            method_row.pack(fill=tk.X, pady=2)
            ttk.Label(method_row, text="Méthode", width=14, anchor=tk.W).pack(side=tk.LEFT)
            method_var = tk.StringVar(value=f.method)
            cb = ttk.Combobox(method_row, textvariable=method_var, width=16,
                              values=["moving_average", "savgol", "gaussian"], state="readonly")
            cb.pack(side=tk.LEFT)
            cb.bind("<<ComboboxSelected>>", lambda _: self._set_attr("method", method_var.get()))
            self._add_int_spinner("Fenêtre", 3, 201, f.window_size, "window_size")
            self._add_int_spinner("Polynôme", 0, 10, f.poly_order, "poly_order")

    def _get_warn_range(self, physio_key: str) -> Optional[Tuple[float, float]]:
        ranges = PHYSIOLOGICAL_RANGES.get(self._channel_type, {})
        entry = ranges.get(physio_key)
        return (entry[0], entry[1]) if entry else None

    def _add_hz_slider(self, label: str, lo: float, hi: float, val: float, attr: str,
                       res: float = 0.1, physio_key: str = "") -> None:
        var = tk.DoubleVar(value=val)
        wr = self._get_warn_range(physio_key) if physio_key else None
        _LabeledScale(self._body, label, lo, hi, resolution=res, variable=var,
                      on_change=lambda: self._set_attr(attr, var.get()),
                      warn_range=wr).pack(fill=tk.X, pady=2)

    def _add_order_spinner(self, label: str, val: int, attr: str) -> None:
        row = ttk.Frame(self._body)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=14, anchor=tk.W).pack(side=tk.LEFT)
        var = tk.IntVar(value=val)
        ttk.Spinbox(row, from_=1, to=12, textvariable=var, width=5,
                     command=lambda: self._set_attr(attr, var.get())).pack(side=tk.LEFT)

    def _add_int_spinner(self, label: str, lo: int, hi: int, val: int, attr: str) -> None:
        row = ttk.Frame(self._body)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=14, anchor=tk.W).pack(side=tk.LEFT)
        var = tk.IntVar(value=val)
        ttk.Spinbox(row, from_=lo, to=hi, textvariable=var, width=5,
                     command=lambda: self._set_attr(attr, var.get())).pack(side=tk.LEFT)

    def _add_type_combo(self, val: str, attr: str) -> None:
        row = ttk.Frame(self._body)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Type IIR", width=14, anchor=tk.W).pack(side=tk.LEFT)
        var = tk.StringVar(value=val)
        cb = ttk.Combobox(row, textvariable=var, width=14,
                          values=["butterworth", "cheby1", "cheby2", "ellip"], state="readonly")
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", lambda _: self._set_attr(attr, var.get()))

    def _add_causal_check(self, val: bool, attr: str) -> None:
        row = ttk.Frame(self._body)
        row.pack(fill=tk.X, pady=2)
        var = tk.BooleanVar(value=val)
        ttk.Checkbutton(row, text="Filtre causal (phase non nulle)", variable=var,
                        command=lambda: self._set_attr(attr, var.get())).pack(side=tk.LEFT)

    def _set_attr(self, attr: str, value: Any) -> None:
        setattr(self.filt, attr, value)
        self._on_change()


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class FilterConfigDialog:
    """Professional filter-configuration dialog (Toplevel) v4.

    Features: dark/light theme, undo/redo, adaptive suggestions, channel search,
    filter tags, 3-panel preview (signal + freq response + PSD), batch apply,
    physiological warnings, import/export, keyboard shortcuts, audit trail,
    favorite presets, per-channel annotations, mini-dashboard, HTML report export.
    """

    def __init__(
        self,
        parent: tk.Tk,
        channel_names: List[str],
        sfreq: float,
        *,
        channel_pipelines: Optional[Dict[str, FilterPipeline]] = None,
        channel_types: Optional[Dict[str, str]] = None,
        signal_getter: Optional[Callable[[str, float, float], np.ndarray]] = None,
        preset_library: Optional[PresetLibrary] = None,
        on_apply: Optional[Callable[[Dict[str, FilterPipeline], bool], None]] = None,
        global_enabled: bool = True,
        audit_log: Optional[FilterAuditLog] = None,
        favorites: Optional[FavoritePresets] = None,
        annotation_store: Optional[ChannelAnnotationStore] = None,
    ):
        self._parent = parent
        self._sfreq = sfreq
        self._channel_names = list(channel_names)
        self._channel_types = channel_types or {}
        self._signal_getter = signal_getter
        self._on_apply_cb = on_apply

        self._pipelines: Dict[str, FilterPipeline] = {}
        for ch in self._channel_names:
            if channel_pipelines and ch in channel_pipelines:
                self._pipelines[ch] = channel_pipelines[ch].deep_copy()
            else:
                self._pipelines[ch] = FilterPipeline()

        self._preset_lib = preset_library or PresetLibrary(_DEFAULT_PRESETS_PATH)
        self._global_enabled = global_enabled
        self._selected_channels: List[str] = []
        self._preview_job: Optional[str] = None
        self._audit = audit_log or FilterAuditLog()
        self._undo = UndoManager(max_depth=60)
        self._suggester = AdaptiveFilterSuggester(self._preset_lib, self._audit)
        self._favorites = favorites or FavoritePresets()
        self._annotations = annotation_store or ChannelAnnotationStore()

        self._theme_name = "dark"
        self._theme = _THEMES["dark"]

        self._undo.save_state(self._pipelines)
        self._build_window()
        # Bloquer jusqu'à fermeture : sinon show_filter_config() reprend le pump Qt
        # alors que grab_set() est actif → PyEval_RestoreThread (Py3.14 + Tk/Qt).
        self._win.protocol("WM_DELETE_WINDOW", self._on_cancel)
        try:
            self._parent.wait_window(self._win)
        except tk.TclError:
            pass

    # -----------------------------------------------------------------------
    # Theme
    # -----------------------------------------------------------------------

    def _apply_theme(self) -> None:
        t = self._theme
        win = self._win
        win.configure(bg=t["bg"])

        style = ttk.Style(win)
        style.theme_use("clam")

        style.configure(".", background=t["bg"], foreground=t["fg"],
                        fieldbackground=t["entry_bg"], bordercolor=t["border"],
                        insertcolor=t["fg"])
        style.configure("TFrame", background=t["bg"])
        style.configure("TLabel", background=t["bg"], foreground=t["fg"])
        style.configure("TLabelframe", background=t["bg"], foreground=t["fg"])
        style.configure("TLabelframe.Label", background=t["bg"], foreground=t["accent"])
        style.configure("TButton", background=t["bg_alt"], foreground=t["fg"])
        style.configure("TCheckbutton", background=t["bg"], foreground=t["fg"])
        style.configure("TEntry", fieldbackground=t["entry_bg"], foreground=t["fg"])
        style.configure("TCombobox", fieldbackground=t["entry_bg"], foreground=t["fg"])
        style.configure("TSpinbox", fieldbackground=t["entry_bg"], foreground=t["fg"])
        style.configure("Horizontal.TScale", background=t["bg"], troughcolor=t["bg_alt"])
        style.configure("TSeparator", background=t["border"])
        style.configure("TPanedwindow", background=t["bg"])

        style.configure("Status.TLabel", background=t["bg_alt"], foreground=t["fg_dim"],
                        padding=(6, 3))
        style.configure("Accent.TButton", background=t["accent"], foreground="#FFFFFF")
        style.configure("Warn.TLabel", foreground=t["warn"])
        style.configure("Error.TLabel", foreground=t["error"])
        style.configure("Ok.TLabel", foreground=t["ok"])

        style.map("TButton",
                  background=[("active", t["accent"])],
                  foreground=[("active", "#FFFFFF")])

    def _toggle_theme(self) -> None:
        self._theme_name = "light" if self._theme_name == "dark" else "dark"
        self._theme = _THEMES[self._theme_name]
        self._apply_theme()
        self._rebuild_pipeline_editor()
        self._schedule_preview()

    # -----------------------------------------------------------------------
    # Window construction
    # -----------------------------------------------------------------------

    def _build_window(self) -> None:
        self._win = tk.Toplevel(self._parent)
        self._win.title("Configuration des Filtres - CESA")
        self._win.geometry("1400x880")
        self._win.minsize(800, 500)
        self._win.resizable(True, True)
        self._win.transient(self._parent)
        self._win.grab_set()

        self._apply_theme()

        self._build_topbar()

        paned = ttk.PanedWindow(self._win, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        left = ttk.Frame(paned, width=240)
        paned.add(left, weight=0)
        self._build_channel_panel(left)

        centre = ttk.Frame(paned, width=420)
        paned.add(centre, weight=1)
        self._centre_frame = centre
        self._pipeline_container: Optional[ttk.Frame] = None

        right = ttk.Frame(paned, width=400)
        paned.add(right, weight=1)
        self._build_preview(right)

        self._build_bottombar()

        self._win.bind("<Control-z>", lambda _: self._do_undo())
        self._win.bind("<Control-y>", lambda _: self._do_redo())
        self._win.bind("<Control-f>", lambda _: self._focus_search())

        if self._channel_names:
            self._ch_listbox.selection_set(0)
            self._on_channel_select()

    # -- top bar -------------------------------------------------------------

    def _build_topbar(self) -> None:
        outer = ttk.Frame(self._win)
        outer.pack(fill=tk.X)

        # -- Row 1: global toggle, preset selector, theme --
        row1 = ttk.Frame(outer, padding=(6, 4, 6, 0))
        row1.pack(fill=tk.X)

        self._global_var = tk.BooleanVar(value=self._global_enabled)
        ttk.Checkbutton(row1, text="Filtrage actif", variable=self._global_var,
                        command=self._schedule_preview).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        ttk.Label(row1, text="Preset :").pack(side=tk.LEFT, padx=(4, 2))
        self._preset_var = tk.StringVar()
        self._preset_combo = ttk.Combobox(row1, textvariable=self._preset_var, width=22, state="readonly")
        self._preset_combo.pack(side=tk.LEFT, padx=(0, 2))
        self._preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        self._preset_filter_var = tk.StringVar(value="Tous")
        ttk.Combobox(row1, textvariable=self._preset_filter_var, width=7,
                     values=["Tous", "eeg", "eog", "emg", "ecg", "generic"],
                     state="readonly").pack(side=tk.LEFT, padx=(0, 4))
        self._preset_filter_var.trace_add("write", lambda *_: self._refresh_preset_list())

        ttk.Button(row1, text="Appliquer", command=self._apply_preset_to_channels).pack(side=tk.LEFT, padx=2)
        self._fav_btn = ttk.Button(row1, text="\u2606", width=3, command=self._toggle_favorite_preset)
        self._fav_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Sauver", command=self._save_current_as_preset).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Suppr.", command=self._delete_preset).pack(side=tk.LEFT, padx=2)

        theme_btn = ttk.Button(row1, text="\u263E", width=3, command=self._toggle_theme)
        theme_btn.pack(side=tk.RIGHT, padx=2)

        # -- Row 2: undo/redo, import/export, suggestions --
        row2 = ttk.Frame(outer, padding=(6, 2, 6, 4))
        row2.pack(fill=tk.X)

        self._undo_btn = ttk.Button(row2, text="\u21A9 Undo", command=self._do_undo, width=7)
        self._undo_btn.pack(side=tk.LEFT, padx=2)
        self._redo_btn = ttk.Button(row2, text="Redo \u21AA", command=self._do_redo, width=7)
        self._redo_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(row2, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        ttk.Button(row2, text="Import", command=self._import_presets).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Export", command=self._export_presets).pack(side=tk.LEFT, padx=2)

        ttk.Separator(row2, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        ttk.Button(row2, text="Suggestions", command=self._show_suggestions).pack(side=tk.LEFT, padx=2)

        self._refresh_preset_list()
        self._update_undo_buttons()

    # -- channel panel (left) ------------------------------------------------

    def _build_channel_panel(self, parent: ttk.Frame) -> None:
        lf = ttk.LabelFrame(parent, text="Canaux", padding=4)
        lf.pack(fill=tk.BOTH, expand=True)

        search_frame = ttk.Frame(lf)
        search_frame.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(search_frame, text="\U0001F50D", width=3).pack(side=tk.LEFT)
        self._search_var = tk.StringVar()
        self._search_entry = ttk.Entry(search_frame, textvariable=self._search_var, width=18)
        self._search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._search_var.trace_add("write", lambda *_: self._filter_channel_list())

        grp_frame = ttk.Frame(lf)
        grp_frame.pack(fill=tk.X, pady=(0, 4))
        for gname in _CHANNEL_GROUPS:
            ttk.Button(grp_frame, text=gname, width=5,
                       command=lambda g=gname: self._select_group(g)).pack(side=tk.LEFT, padx=1)
        ttk.Button(grp_frame, text="\u2205", width=3, command=self._deselect_all).pack(side=tk.LEFT, padx=1)

        list_frame = ttk.Frame(lf)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self._ch_listbox = tk.Listbox(
            list_frame, selectmode=tk.EXTENDED, activestyle="dotbox",
            font=("Segoe UI", 9), exportselection=False,
            bg=self._theme["bg_alt"], fg=self._theme["fg"],
            selectbackground=self._theme["accent"], selectforeground="#FFFFFF",
            highlightthickness=0, relief=tk.FLAT,
        )
        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._ch_listbox.yview)
        self._ch_listbox.configure(yscrollcommand=sb.set)
        self._ch_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._visible_indices: List[int] = list(range(len(self._channel_names)))
        self._populate_channel_listbox()
        self._ch_listbox.bind("<<ListboxSelect>>", self._on_channel_select)

        batch_frame = ttk.LabelFrame(lf, text="Batch", padding=4)
        batch_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(batch_frame, text="Copier vers sélection",
                   command=self._batch_copy_pipeline).pack(fill=tk.X, pady=1)
        ttk.Button(batch_frame, text="Appliquer preset",
                   command=self._batch_apply_preset).pack(fill=tk.X, pady=1)
        ttk.Button(batch_frame, text="Réinitialiser",
                   command=self._batch_reset).pack(fill=tk.X, pady=1)

    def _populate_channel_listbox(self) -> None:
        self._ch_listbox.delete(0, tk.END)
        for i in self._visible_indices:
            ch = self._channel_names[i]
            ctype = self._channel_types.get(ch, "?").upper()[:3]
            pl = self._pipelines.get(ch)
            tags = ""
            if pl and pl.enabled and pl.filters:
                active = [f for f in pl.filters if f.enabled]
                tags = " ".join(_short_filter_label(f) for f in active[:3])
                if len(active) > 3:
                    tags += f" +{len(active) - 3}"
            status = tags if tags else "---"
            self._ch_listbox.insert(tk.END, f"[{ctype}] {ch}  | {status}")

    def _filter_channel_list(self) -> None:
        query = self._search_var.get().strip().lower()
        prev_sel_names = set(self._selected_channels)
        if query:
            self._visible_indices = [
                i for i, ch in enumerate(self._channel_names)
                if query in ch.lower() or query in self._channel_types.get(ch, "").lower()
            ]
        else:
            self._visible_indices = list(range(len(self._channel_names)))
        self._populate_channel_listbox()
        for li, real_i in enumerate(self._visible_indices):
            if self._channel_names[real_i] in prev_sel_names:
                self._ch_listbox.selection_set(li)

    def _focus_search(self) -> None:
        self._search_entry.focus_set()

    def _select_group(self, group: str) -> None:
        self._ch_listbox.selection_clear(0, tk.END)
        types = _CHANNEL_GROUPS[group]
        for li, real_i in enumerate(self._visible_indices):
            ct = self._channel_types.get(self._channel_names[real_i], "unknown").lower()
            if not types or ct in types:
                self._ch_listbox.selection_set(li)
        self._on_channel_select()

    def _deselect_all(self) -> None:
        self._ch_listbox.selection_clear(0, tk.END)
        self._selected_channels = []
        self._rebuild_pipeline_editor()

    def _on_channel_select(self, _event: Any = None) -> None:
        indices = list(self._ch_listbox.curselection())
        self._selected_channels = [
            self._channel_names[self._visible_indices[i]]
            for i in indices
            if 0 <= i < len(self._visible_indices)
        ]
        if self._selected_channels:
            self._rebuild_pipeline_editor()
            self._schedule_preview()
        self._update_status()

    def _refresh_channel_listbox(self) -> None:
        prev_sel_names = set(self._selected_channels)
        self._populate_channel_listbox()
        for li, real_i in enumerate(self._visible_indices):
            if self._channel_names[real_i] in prev_sel_names:
                self._ch_listbox.selection_set(li)

    # -- pipeline editor (centre panel) --------------------------------------

    def _primary_channel(self) -> Optional[str]:
        return self._selected_channels[0] if self._selected_channels else None

    def _save_undo_snapshot(self) -> None:
        self._undo.save_state(self._pipelines)
        self._update_undo_buttons()

    def _rebuild_pipeline_editor(self) -> None:
        if self._pipeline_container:
            self._pipeline_container.destroy()

        container = ttk.Frame(self._centre_frame)
        container.pack(fill=tk.BOTH, expand=True)
        self._pipeline_container = container

        ch = self._primary_channel()
        if not ch:
            ttk.Label(container, text="Sélectionnez un ou plusieurs canaux",
                      font=("Segoe UI", 11)).pack(pady=20)
            return

        pipeline = self._pipelines.get(ch, FilterPipeline())
        ch_type = self._channel_types.get(ch, "generic")

        hdr = ttk.Frame(container)
        hdr.pack(fill=tk.X, pady=(4, 4))

        n_sel = len(self._selected_channels)
        title = f"Pipeline : {ch}" if n_sel == 1 else f"Pipeline : {ch} (+{n_sel - 1})"
        ttk.Label(hdr, text=title, font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)

        pipe_enabled_var = tk.BooleanVar(value=pipeline.enabled)

        def _toggle_pipe():
            self._save_undo_snapshot()
            pipeline.enabled = pipe_enabled_var.get()
            self._audit.record(ch, "toggle_pipeline", enabled=pipeline.enabled)
            self._refresh_channel_listbox()
            self._schedule_preview()
            self._update_status()

        ttk.Checkbutton(hdr, text="Pipeline actif", variable=pipe_enabled_var,
                        command=_toggle_pipe).pack(side=tk.RIGHT)

        ann_frame = ttk.LabelFrame(container, text="Annotation", padding=4)
        ann_frame.pack(fill=tk.X, padx=4, pady=(2, 4))
        ann_var = tk.StringVar(value=self._annotations.get_text(ch))
        ann_entry = ttk.Entry(ann_frame, textvariable=ann_var, font=("Segoe UI", 9))
        ann_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        def _save_annotation(_event=None):
            self._annotations.set(ch, ann_var.get().strip())
            self._audit.record(ch, "set_annotation", text=ann_var.get().strip())
        ann_entry.bind("<Return>", _save_annotation)
        ann_entry.bind("<FocusOut>", _save_annotation)
        ttk.Button(ann_frame, text="\u2714", width=3, command=_save_annotation).pack(side=tk.LEFT)

        warnings = pipeline.physiological_warnings(ch_type, self._sfreq)
        errs = pipeline.validate(self._sfreq)

        if warnings or errs:
            alert_frame = ttk.Frame(container)
            alert_frame.pack(fill=tk.X, padx=4, pady=2)
            for e in errs[:3]:
                ttk.Label(alert_frame, text=f"\u274C {e}", style="Error.TLabel",
                          wraplength=380).pack(anchor=tk.W)
            for w in warnings[:5]:
                ttk.Label(alert_frame, text=f"\u26A0 {w}", style="Warn.TLabel",
                          wraplength=380, font=("Segoe UI", 8)).pack(anchor=tk.W)

        canvas = tk.Canvas(container, highlightthickness=0, bg=self._theme["bg"])
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        cards_frame = ttk.Frame(canvas)
        cards_frame.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=cards_frame, anchor="nw",
                             tags="cards_window")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Stretch the inner frame to fill canvas width on resize
        def _resize_cards_inner(event):
            canvas.itemconfigure("cards_window", width=event.width)
        canvas.bind("<Configure>", _resize_cards_inner)

        def _mw(event):
            try:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except Exception:
                pass
        canvas.bind("<MouseWheel>", _mw)
        cards_frame.bind("<MouseWheel>", _mw)

        def _bind_mw_recursive(widget):
            """Bind mousewheel on all children so scroll works everywhere."""
            widget.bind("<MouseWheel>", _mw)
            for child in widget.winfo_children():
                _bind_mw_recursive(child)

        for i, filt in enumerate(pipeline.filters):
            card = _FilterCard(
                cards_frame, filt, i,
                channel_type=ch_type,
                on_change=lambda: self._on_filter_param_changed(),
                on_delete=lambda idx=i: self._delete_filter(idx),
                on_move_up=lambda idx=i: self._move_filter(idx, idx - 1),
                on_move_down=lambda idx=i: self._move_filter(idx, idx + 1),
            )
            card.pack(fill=tk.X, padx=4, pady=4)
            _bind_mw_recursive(card)

        add_frame = ttk.Frame(cards_frame)
        add_frame.pack(fill=tk.X, padx=4, pady=8)
        self._add_type_var = tk.StringVar(value=list(_FILTER_TYPES.keys())[0])
        ttk.Combobox(add_frame, textvariable=self._add_type_var, width=18,
                     values=list(_FILTER_TYPES.keys()), state="readonly").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(add_frame, text="+ Ajouter filtre", command=self._add_filter).pack(side=tk.LEFT)

    def _on_filter_param_changed(self) -> None:
        ch = self._primary_channel()
        if ch:
            pipeline = self._pipelines.get(ch)
            if pipeline:
                self._audit.record(ch, "param_changed",
                                   pipeline_summary=[f.to_dict() for f in pipeline.filters])
        self._schedule_preview()
        self._refresh_channel_listbox()
        self._rebuild_pipeline_editor()
        self._update_status()

    def _add_filter(self) -> None:
        ch = self._primary_channel()
        if not ch:
            return
        self._save_undo_snapshot()
        type_label = self._add_type_var.get()
        cls = _FILTER_TYPES.get(type_label)
        if cls is None:
            return
        pipeline = self._pipelines.setdefault(ch, FilterPipeline())
        new_filt = cls()
        pipeline.filters.append(new_filt)
        self._audit.record(ch, "add_filter", filter_type=type(new_filt).__name__)
        self._rebuild_pipeline_editor()
        self._refresh_channel_listbox()
        self._schedule_preview()

    def _delete_filter(self, idx: int) -> None:
        ch = self._primary_channel()
        if not ch:
            return
        self._save_undo_snapshot()
        pipeline = self._pipelines.get(ch)
        if pipeline and 0 <= idx < len(pipeline.filters):
            removed = pipeline.filters[idx]
            self._audit.record(ch, "delete_filter",
                               filter_type=type(removed).__name__, index=idx)
            pipeline.remove(idx)
        self._rebuild_pipeline_editor()
        self._refresh_channel_listbox()
        self._schedule_preview()

    def _move_filter(self, from_idx: int, to_idx: int) -> None:
        ch = self._primary_channel()
        if not ch:
            return
        self._save_undo_snapshot()
        pipeline = self._pipelines.get(ch)
        if pipeline:
            pipeline.move(from_idx, to_idx)
            self._audit.record(ch, "move_filter", from_idx=from_idx, to_idx=to_idx)
        self._rebuild_pipeline_editor()
        self._schedule_preview()

    # -- undo / redo ---------------------------------------------------------

    def _do_undo(self) -> None:
        restored = self._undo.undo(self._pipelines)
        if restored is not None:
            self._pipelines = restored
            self._audit.record("*", "undo")
            self._rebuild_pipeline_editor()
            self._refresh_channel_listbox()
            self._schedule_preview()
            self._update_undo_buttons()
            self._update_status()

    def _do_redo(self) -> None:
        restored = self._undo.redo(self._pipelines)
        if restored is not None:
            self._pipelines = restored
            self._audit.record("*", "redo")
            self._rebuild_pipeline_editor()
            self._refresh_channel_listbox()
            self._schedule_preview()
            self._update_undo_buttons()
            self._update_status()

    def _update_undo_buttons(self) -> None:
        try:
            self._undo_btn.configure(
                state=tk.NORMAL if self._undo.can_undo else tk.DISABLED)
            self._redo_btn.configure(
                state=tk.NORMAL if self._undo.can_redo else tk.DISABLED)
        except Exception:
            pass

    # -- preview (right panel) -----------------------------------------------

    def _build_preview(self, parent: ttk.Frame) -> None:
        lf = ttk.LabelFrame(parent, text="Aperçu temps réel", padding=4)
        lf.pack(fill=tk.BOTH, expand=True)

        # Scrollable wrapper so the preview adapts when the window is small
        preview_canvas = tk.Canvas(lf, highlightthickness=0)
        preview_scroll = ttk.Scrollbar(lf, orient=tk.VERTICAL, command=preview_canvas.yview)
        preview_inner = ttk.Frame(preview_canvas)
        preview_inner.bind(
            "<Configure>",
            lambda _: preview_canvas.configure(scrollregion=preview_canvas.bbox("all")),
        )
        preview_canvas.create_window((0, 0), window=preview_inner, anchor="nw")
        preview_canvas.configure(yscrollcommand=preview_scroll.set)
        preview_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preview_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        def _preview_mousewheel(event):
            try:
                preview_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except Exception:
                pass
        preview_canvas.bind("<MouseWheel>", _preview_mousewheel)
        preview_inner.bind("<MouseWheel>", _preview_mousewheel)

        try:
            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure

            t = self._theme
            self._fig = Figure(figsize=(4.5, 7), dpi=90,
                               facecolor=t["plot_bg"])
            gs = self._fig.add_gridspec(3, 1, hspace=0.45)
            self._ax_signal = self._fig.add_subplot(gs[0])
            self._ax_freq = self._fig.add_subplot(gs[1])
            self._ax_psd = self._fig.add_subplot(gs[2])

            for ax in (self._ax_signal, self._ax_freq, self._ax_psd):
                ax.set_facecolor(t["plot_bg"])
                ax.tick_params(colors=t["plot_fg"], labelsize=7)
                for spine in ax.spines.values():
                    spine.set_color(t["plot_grid"])

            self._canvas_mpl = FigureCanvasTkAgg(self._fig, master=preview_inner)
            mpl_widget = self._canvas_mpl.get_tk_widget()
            mpl_widget.configure(width=380, height=620)
            mpl_widget.pack(fill=tk.BOTH, expand=True)

            self._snr_label = ttk.Label(preview_inner, text="SNR: ---", style="Status.TLabel")
            self._snr_label.pack(fill=tk.X)

            self._has_preview = True
        except Exception:
            ttk.Label(preview_inner, text="(matplotlib non disponible)").pack()
            self._has_preview = False

    def _schedule_preview(self) -> None:
        if not self._has_preview:
            return
        if self._preview_job is not None:
            try:
                self._win.after_cancel(self._preview_job)
            except Exception:
                pass
        self._preview_job = self._win.after(_PREVIEW_DEBOUNCE_MS, self._update_preview)

    def _update_preview(self) -> None:
        self._preview_job = None
        ch = self._primary_channel()
        if not ch:
            return
        pipeline = self._pipelines.get(ch, FilterPipeline())
        t = self._theme
        duration_s = 5.0

        snippet: Optional[np.ndarray] = None
        if self._signal_getter is not None:
            try:
                snippet = self._signal_getter(ch, 0.0, duration_s)
            except Exception:
                snippet = None
        if snippet is None or len(snippet) == 0:
            n_pts = int(self._sfreq * duration_s)
            tt = np.arange(n_pts) / self._sfreq
            snippet = (
                30 * np.sin(2 * np.pi * 10 * tt)
                + 10 * np.sin(2 * np.pi * 0.5 * tt)
                + 5 * np.sin(2 * np.pi * 50 * tt)
                + np.random.default_rng(42).normal(0, 3, n_pts)
            )

        t_arr = np.arange(len(snippet)) / self._sfreq
        filtered = None
        if pipeline.enabled and pipeline.filters and self._global_var.get():
            try:
                filtered = pipeline.apply(snippet, self._sfreq)
            except Exception:
                filtered = None

        # --- Signal overlay ---
        ax = self._ax_signal
        ax.clear()
        ax.plot(t_arr, snippet, color=t["plot_raw"], linewidth=0.6, label="Brut", alpha=0.6)
        if filtered is not None:
            ax.plot(t_arr, filtered, color=t["plot_filt"], linewidth=1.0, label="Filtré")
            # artifact indicator: highlight samples where |value| > 5*std
            std = np.std(filtered)
            if std > 0:
                artifact_mask = np.abs(filtered) > 5 * std
                if np.any(artifact_mask):
                    ax.fill_between(t_arr, ax.get_ylim()[0], ax.get_ylim()[1],
                                    where=artifact_mask, alpha=0.15, color=t["error"],
                                    label="Artefact?")
        ax.set_title(f"Signal : {ch} ({duration_s:.0f} s)", fontsize=9, color=t["plot_fg"])
        ax.set_xlabel("Temps (s)", fontsize=8, color=t["plot_fg"])
        ax.set_ylabel("\u00b5V", fontsize=8, color=t["plot_fg"])
        ax.legend(fontsize=7, loc="upper right", facecolor=t["bg_alt"], edgecolor=t["border"],
                  labelcolor=t["plot_fg"])

        # --- Frequency response ---
        ax2 = self._ax_freq
        ax2.clear()
        try:
            freqs, mag_db = pipeline.frequency_response(self._sfreq, n_points=512)
            ax2.plot(freqs, mag_db, color=t["plot_freq"], linewidth=1.0)
            ax2.set_xlim(0, min(self._sfreq / 2, 150))
            ax2.set_ylim(-60, 5)
            ax2.axhline(-3, color=t["error"], linestyle="--", linewidth=0.5, label="-3 dB")
            ax2.legend(fontsize=7, facecolor=t["bg_alt"], edgecolor=t["border"],
                       labelcolor=t["plot_fg"])
        except Exception:
            pass
        ax2.set_title("Réponse en fréquence", fontsize=9, color=t["plot_fg"])
        ax2.set_xlabel("Fréquence (Hz)", fontsize=8, color=t["plot_fg"])
        ax2.set_ylabel("dB", fontsize=8, color=t["plot_fg"])

        # --- PSD comparison ---
        ax3 = self._ax_psd
        ax3.clear()
        try:
            n = len(snippet)
            freqs_psd = np.fft.rfftfreq(n, 1.0 / self._sfreq)
            psd_raw = 10 * np.log10(np.abs(np.fft.rfft(snippet)) ** 2 + 1e-12)
            ax3.plot(freqs_psd, psd_raw, color=t["plot_raw"], linewidth=0.6, label="Brut", alpha=0.6)
            if filtered is not None:
                psd_filt = 10 * np.log10(np.abs(np.fft.rfft(filtered)) ** 2 + 1e-12)
                ax3.plot(freqs_psd, psd_filt, color=t["plot_psd"], linewidth=1.0, label="Filtré")
            ax3.set_xlim(0, min(self._sfreq / 2, 80))
            ax3.legend(fontsize=7, facecolor=t["bg_alt"], edgecolor=t["border"],
                       labelcolor=t["plot_fg"])
        except Exception:
            pass
        ax3.set_title("Densité spectrale (PSD)", fontsize=9, color=t["plot_fg"])
        ax3.set_xlabel("Fréquence (Hz)", fontsize=8, color=t["plot_fg"])
        ax3.set_ylabel("dB", fontsize=8, color=t["plot_fg"])

        for a in (ax, ax2, ax3):
            a.set_facecolor(t["plot_bg"])
            a.tick_params(colors=t["plot_fg"], labelsize=7)
            for spine in a.spines.values():
                spine.set_color(t["plot_grid"])

        self._fig.set_facecolor(t["plot_bg"])
        self._fig.tight_layout(pad=1.5)
        try:
            self._canvas_mpl.draw_idle()
        except Exception:
            pass

        # SNR indicator
        if filtered is not None:
            noise = snippet - filtered
            p_sig = np.mean(filtered ** 2) + 1e-12
            p_noise = np.mean(noise ** 2) + 1e-12
            snr_db = 10 * np.log10(p_sig / p_noise)
            self._snr_label.configure(text=f"SNR: {snr_db:.1f} dB")
        else:
            self._snr_label.configure(text="SNR: ---")

    # -- adaptive suggestions ------------------------------------------------

    def _show_suggestions(self) -> None:
        ch = self._primary_channel()
        if not ch:
            messagebox.showinfo("Suggestions", "Sélectionnez un canal d'abord.")
            return

        ch_type = self._channel_types.get(ch, "generic")
        snippet = None
        if self._signal_getter:
            try:
                snippet = self._signal_getter(ch, 0.0, 10.0)
            except Exception:
                pass

        suggestions = self._suggester.suggest_for_channel(
            ch_type, context="default", signal_snippet=snippet, sfreq=self._sfreq,
        )
        if not suggestions:
            messagebox.showinfo("Suggestions", "Aucune suggestion disponible pour ce canal.")
            return

        win = tk.Toplevel(self._win)
        win.title(f"Suggestions - {ch}")
        win.geometry("520x380")
        win.transient(self._win)
        win.grab_set()
        win.configure(bg=self._theme["bg"])

        ttk.Label(win, text=f"Suggestions pour {ch} [{ch_type.upper()}]",
                  font=("Segoe UI", 12, "bold")).pack(pady=(12, 8))

        for sg in suggestions:
            row = ttk.Frame(win, padding=4)
            row.pack(fill=tk.X, padx=12, pady=2)

            conf_pct = f"{sg.confidence * 100:.0f}%"
            ttk.Label(row, text=f"\u2022 {sg.preset_name}", font=("Segoe UI", 10, "bold")).pack(
                side=tk.LEFT)
            ttk.Label(row, text=f"  [{conf_pct}]", style="Ok.TLabel").pack(side=tk.LEFT)
            ttk.Button(row, text="Appliquer", width=9,
                       command=lambda s=sg: self._accept_suggestion(s, ch, win)).pack(side=tk.RIGHT)

            reason_lbl = ttk.Label(win, text=f"     {sg.reason}",
                                   wraplength=460, font=("Segoe UI", 8))
            reason_lbl.pack(anchor=tk.W, padx=16)

        ttk.Button(win, text="Fermer", command=win.destroy).pack(pady=12)

    def _accept_suggestion(self, sg: FilterSuggestion, ch: str,
                           popup: tk.Toplevel) -> None:
        self._save_undo_snapshot()
        self._suggester.accept_suggestion(sg, ch, self._pipelines)
        self._rebuild_pipeline_editor()
        self._refresh_channel_listbox()
        self._schedule_preview()
        self._update_status()
        popup.destroy()

    # -- presets -------------------------------------------------------------

    def _toggle_favorite_preset(self) -> None:
        name = self._preset_var.get()
        if not name:
            return
        is_fav = self._favorites.toggle(name)
        self._audit.record("*", "toggle_favorite", preset_name=name, is_favorite=is_fav)
        self._refresh_preset_list()
        self._update_fav_button()

    def _update_fav_button(self) -> None:
        name = self._preset_var.get()
        try:
            if name and self._favorites.is_favorite(name):
                self._fav_btn.configure(text="\u2605")
            else:
                self._fav_btn.configure(text="\u2606")
        except Exception:
            pass

    def _refresh_preset_list(self) -> None:
        pf = self._preset_filter_var.get()
        names = self._preset_lib.list_names() if pf == "Tous" else self._preset_lib.list_names(pf)
        fav_names = self._favorites.names
        display: List[str] = []
        for n in names:
            display.append(f"\u2605 {n}" if n in fav_names else n)
        sorted_display = sorted(display, key=lambda d: (0 if d.startswith("\u2605") else 1, d))
        raw_sorted = [d.lstrip("\u2605 ") for d in sorted_display]
        self._preset_combo["values"] = raw_sorted
        if raw_sorted:
            self._preset_var.set(raw_sorted[0])
        self._update_fav_button()

    def _on_preset_selected(self, _event: Any = None) -> None:
        self._update_fav_button()

    def _apply_preset_to_channels(self) -> None:
        name = self._preset_var.get()
        preset = self._preset_lib.get(name)
        if preset is None:
            messagebox.showwarning("Preset", f"Preset '{name}' introuvable.")
            return
        self._save_undo_snapshot()
        targets = self._selected_channels or ([self._primary_channel()] if self._primary_channel() else [])
        for ch in targets:
            self._pipelines[ch] = preset.pipeline.deep_copy()
            self._audit.record(ch, "apply_preset", preset_name=name)
        self._rebuild_pipeline_editor()
        self._refresh_channel_listbox()
        self._schedule_preview()
        self._update_status()
        if len(targets) > 1:
            messagebox.showinfo("Preset", f"Preset '{name}' appliqué à {len(targets)} canaux.")

    def _save_current_as_preset(self) -> None:
        ch = self._primary_channel()
        if not ch:
            return
        pipeline = self._pipelines.get(ch, FilterPipeline())
        win = tk.Toplevel(self._win)
        win.title("Sauvegarder le preset")
        win.geometry("400x260")
        win.transient(self._win)
        win.grab_set()
        win.configure(bg=self._theme["bg"])

        ttk.Label(win, text="Nom du preset :").pack(pady=(12, 4))
        name_var = tk.StringVar(value=f"Custom - {ch}")
        ttk.Entry(win, textvariable=name_var, width=32).pack()
        ttk.Label(win, text="Description :").pack(pady=(8, 4))
        desc_var = tk.StringVar()
        ttk.Entry(win, textvariable=desc_var, width=32).pack()
        ttk.Label(win, text="Type de canal :").pack(pady=(8, 4))
        ctype_var = tk.StringVar(value=self._channel_types.get(ch, "generic"))
        ttk.Combobox(win, textvariable=ctype_var, values=["eeg", "eog", "emg", "ecg", "generic"],
                     width=14, state="readonly").pack()

        def _save():
            n = name_var.get().strip()
            if not n:
                messagebox.showwarning("Erreur", "Le nom ne peut pas être vide.", parent=win)
                return
            p = FilterPreset(name=n, description=desc_var.get(),
                             channel_type=ctype_var.get(), pipeline=pipeline.deep_copy(),
                             builtin=False)
            try:
                self._preset_lib.add(p, overwrite=True)
                self._preset_lib.save_user()
                self._audit.record(ch, "save_preset", preset_name=n)
                self._refresh_preset_list()
                win.destroy()
            except Exception as exc:
                messagebox.showerror("Erreur", str(exc), parent=win)

        ttk.Button(win, text="Sauvegarder", command=_save).pack(pady=12)

    def _delete_preset(self) -> None:
        name = self._preset_var.get()
        if not name:
            return
        preset = self._preset_lib.get(name)
        if preset and preset.builtin:
            messagebox.showwarning("Preset", "Impossible de supprimer un preset intégré.")
            return
        if not messagebox.askyesno("Supprimer", f"Supprimer le preset '{name}' ?"):
            return
        try:
            self._preset_lib.remove(name)
            self._preset_lib.save_user()
            self._audit.record("*", "delete_preset", preset_name=name)
            self._refresh_preset_list()
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))

    def _import_presets(self) -> None:
        path = filedialog.askopenfilename(
            title="Importer des presets",
            filetypes=[("JSON", "*.json"), ("Tous", "*.*")],
            parent=self._win)
        if not path:
            return
        try:
            count = self._preset_lib.import_presets(path, overwrite=False)
            self._preset_lib.save_user()
            self._refresh_preset_list()
            self._audit.record("*", "import_presets", path=path, count=count)
            messagebox.showinfo("Import", f"{count} preset(s) importé(s).")
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))

    def _export_presets(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Exporter des presets", defaultextension=".json",
            filetypes=[("JSON", "*.json")], parent=self._win)
        if not path:
            return
        try:
            count = self._preset_lib.export_presets(path)
            self._audit.record("*", "export_presets", path=path, count=count)
            messagebox.showinfo("Export", f"{count} preset(s) exporté(s).")
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))

    # -- batch operations ----------------------------------------------------

    def _batch_copy_pipeline(self) -> None:
        if len(self._selected_channels) < 2:
            messagebox.showinfo("Batch", "Sélectionnez au moins 2 canaux.")
            return
        self._save_undo_snapshot()
        src = self._selected_channels[0]
        src_pipe = self._pipelines.get(src, FilterPipeline())
        count = 0
        for ch in self._selected_channels[1:]:
            self._pipelines[ch] = src_pipe.deep_copy()
            self._audit.record(ch, "batch_copy", source=src)
            count += 1
        self._refresh_channel_listbox()
        self._update_status()
        messagebox.showinfo("Batch", f"Pipeline copié vers {count} canal(aux).")

    def _batch_apply_preset(self) -> None:
        self._apply_preset_to_channels()

    def _batch_reset(self) -> None:
        if not self._selected_channels:
            return
        self._save_undo_snapshot()
        for ch in self._selected_channels:
            self._pipelines[ch] = FilterPipeline()
            self._audit.record(ch, "batch_reset")
        self._rebuild_pipeline_editor()
        self._refresh_channel_listbox()
        self._schedule_preview()
        self._update_status()
        messagebox.showinfo("Batch", f"{len(self._selected_channels)} canal(aux) réinitialisé(s).")

    # -- bottom bar + status -------------------------------------------------

    def _build_bottombar(self) -> None:
        bar = ttk.Frame(self._win, padding=6)
        bar.pack(fill=tk.X)

        self._status_label = ttk.Label(bar, text="", style="Status.TLabel")
        self._status_label.pack(side=tk.LEFT, padx=(0, 8))

        # Secondary actions on the left side
        ttk.Button(bar, text="Dashboard", command=self._show_dashboard).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Audit...", command=self._export_audit).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Rapport HTML", command=self._export_html_report).pack(side=tk.LEFT, padx=2)

        # Primary actions on the right side (always visible)
        ttk.Button(bar, text="Appliquer et Fermer", style="Accent.TButton",
                   command=self._on_apply).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bar, text="Annuler", command=self._on_cancel).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bar, text="Réinitialiser tout", command=self._on_reset).pack(side=tk.RIGHT, padx=4)

        self._update_status()

    def _update_status(self) -> None:
        active = sum(1 for p in self._pipelines.values() if p.enabled and p.filters)
        total = len(self._pipelines)
        n_warn = sum(
            len(self._pipelines[ch].physiological_warnings(
                self._channel_types.get(ch, "generic"), self._sfreq))
            for ch in self._channel_names
            if ch in self._pipelines and self._pipelines[ch].enabled
        )
        sel = len(self._selected_channels)
        parts = [
            f"{active}/{total} canaux filtrés",
            f"{sel} sélectionné(s)",
            f"Undo: {self._undo.undo_depth}",
        ]
        if n_warn > 0:
            parts.append(f"\u26A0 {n_warn} avert.")
        try:
            self._status_label.configure(text="  |  ".join(parts))
        except Exception:
            pass

    def _on_apply(self) -> None:
        self._audit.record("*", "apply_all", channel_count=len(self._pipelines))
        if self._on_apply_cb:
            self._on_apply_cb(self._pipelines, self._global_var.get())
        self._win.destroy()

    def _on_reset(self) -> None:
        self._save_undo_snapshot()
        for ch in self._channel_names:
            self._pipelines[ch] = FilterPipeline()
        self._global_var.set(True)
        self._audit.record("*", "reset_all")
        self._rebuild_pipeline_editor()
        self._refresh_channel_listbox()
        self._schedule_preview()
        self._update_status()

    def _on_cancel(self) -> None:
        self._win.destroy()

    def _export_audit(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Exporter le journal d'audit", defaultextension=".json",
            filetypes=[("JSON", "*.json")], parent=self._win)
        if not path:
            return
        try:
            self._audit.export_json(path)
            messagebox.showinfo("Audit", f"Journal exporté ({len(self._audit.entries)} entrées).")
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))

    # -- HTML report export ---------------------------------------------------

    def _export_html_report(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Exporter rapport HTML", defaultextension=".html",
            filetypes=[("HTML", "*.html")], parent=self._win)
        if not path:
            return
        try:
            from CESA.report_generator import ReportConfig, ReportGenerator

            raw_data: Dict[str, np.ndarray] = {}
            for ch in self._channel_names:
                if self._signal_getter:
                    try:
                        raw_data[ch] = self._signal_getter(ch, 0.0, 10.0)
                    except Exception:
                        pass

            gen = ReportGenerator(
                pipelines=self._pipelines,
                audit_log=self._audit,
                annotations=self._annotations.as_text_dict(),
                raw_data=raw_data,
                sfreq=self._sfreq,
                config=ReportConfig(title="CESA Filter Report"),
            )
            gen.generate(path)
            self._audit.record("*", "export_html_report", path=path)
            messagebox.showinfo("Rapport", f"Rapport HTML exporté : {path}")
        except Exception as exc:
            messagebox.showerror("Erreur", f"Export rapport échoué : {exc}")

    # -- mini-dashboard popup -------------------------------------------------

    def _show_dashboard(self) -> None:
        win = tk.Toplevel(self._win)
        win.title("Dashboard - CESA Filtrage")
        win.geometry("560x500")
        win.transient(self._win)
        win.configure(bg=self._theme["bg"])

        ttk.Label(win, text="Mini-Dashboard Filtrage",
                  font=("Segoe UI", 14, "bold")).pack(pady=(12, 8))

        frame = ttk.Frame(win, padding=8)
        frame.pack(fill=tk.BOTH, expand=True)

        total = len(self._channel_names)
        active = sum(1 for p in self._pipelines.values() if p.enabled and p.filters)
        n_warn = sum(
            len(self._pipelines[ch].physiological_warnings(
                self._channel_types.get(ch, "generic"), self._sfreq))
            for ch in self._channel_names
            if ch in self._pipelines and self._pipelines[ch].enabled
        )
        n_annotations = sum(1 for ch in self._channel_names if self._annotations.get_text(ch))
        n_audit = len(self._audit.entries)
        n_fav = len(self._favorites.names)

        rows = [
            ("Canaux totaux", str(total)),
            ("Canaux avec filtres actifs", str(active)),
            ("Avertissements physiologiques", str(n_warn)),
            ("Annotations canal", str(n_annotations)),
            ("Entrées audit", str(n_audit)),
            ("Presets favoris", str(n_fav)),
            ("Profondeur Undo", str(self._undo.undo_depth)),
            ("Profondeur Redo", str(self._undo.redo_depth)),
        ]

        for i, (label, val) in enumerate(rows):
            ttk.Label(frame, text=label, font=("Segoe UI", 10)).grid(
                row=i, column=0, sticky=tk.W, padx=(0, 12), pady=2)
            style = "Warn.TLabel" if "avert" in label.lower() and int(val) > 0 else "Ok.TLabel"
            ttk.Label(frame, text=val, font=("Segoe UI", 10, "bold"),
                      style=style).grid(row=i, column=1, sticky=tk.E, pady=2)

        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(
            row=len(rows), column=0, columnspan=2, sticky="ew", pady=8)

        snr_lf = ttk.LabelFrame(frame, text="SNR moyen par canal (filtrage actif)", padding=4)
        snr_lf.grid(row=len(rows) + 1, column=0, columnspan=2, sticky="ew")

        canvas = tk.Canvas(snr_lf, height=160, bg=self._theme["bg_alt"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(snr_lf, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for ch in self._channel_names:
            pipeline = self._pipelines.get(ch, FilterPipeline())
            if not (pipeline.enabled and pipeline.filters):
                continue
            snippet = None
            if self._signal_getter:
                try:
                    snippet = self._signal_getter(ch, 0.0, 5.0)
                except Exception:
                    pass
            if snippet is None or len(snippet) == 0:
                continue
            try:
                filtered = pipeline.apply(snippet, self._sfreq)
                noise = snippet - filtered
                p_sig = np.mean(filtered ** 2) + 1e-12
                p_noise = np.mean(noise ** 2) + 1e-12
                snr_db = 10 * np.log10(p_sig / p_noise)
                snr_text = f"{snr_db:.1f} dB"
            except Exception:
                snr_text = "N/A"
            row = ttk.Frame(inner)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=ch, width=20, anchor=tk.W).pack(side=tk.LEFT)
            ttk.Label(row, text=snr_text, width=12, anchor=tk.E).pack(side=tk.RIGHT)

        recent_lf = ttk.LabelFrame(frame, text="Dernières actions (audit)", padding=4)
        recent_lf.grid(row=len(rows) + 2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        recent = self._audit.entries[-8:] if self._audit.entries else []
        for e in reversed(recent):
            txt = f"[{e.timestamp[-8:]}] {e.channel} : {e.action}"
            ttk.Label(recent_lf, text=txt, font=("Segoe UI", 8),
                      wraplength=500).pack(anchor=tk.W)

        ttk.Button(win, text="Fermer", command=win.destroy).pack(pady=12)

    # -- public accessors ----------------------------------------------------

    @property
    def audit_log(self) -> FilterAuditLog:
        return self._audit

    @property
    def pipelines(self) -> Dict[str, FilterPipeline]:
        return self._pipelines

    @property
    def undo_manager(self) -> UndoManager:
        return self._undo

    @property
    def favorites(self) -> FavoritePresets:
        return self._favorites

    @property
    def annotation_store(self) -> ChannelAnnotationStore:
        return self._annotations
