"""Dark / light theme palettes for the PyQtGraph viewer.

Clinical sleep-stage colours follow a perceptual gradient:
  Wake = warm yellow (active)
  N1   = light blue (transition)
  N2   = medium blue (stable sleep)
  N3   = deep blue (slow-wave)
  REM  = pink/magenta (high cortical activity)
  U    = neutral grey (unknown / artefact)
"""

from __future__ import annotations

from typing import Dict

ThemePalette = Dict[str, str]

DARK: ThemePalette = {
    # -- General UI --
    "background": "#1E1E2E",
    "foreground": "#CDD6F4",
    "grid": "#45475A",
    "border": "#585B70",
    "surface": "#313244",
    "surface_alt": "#2A2A3C",
    "accent": "#89B4FA",
    "accent_dim": "#45475A",
    "text_dim": "#6C7086",

    # -- Signal colours --
    "eeg_color": "#89B4FA",
    "eog_color": "#A6E3A1",
    "emg_color": "#F9E2AF",
    "ecg_color": "#F38BA8",
    "highlight": "#89B4FA",

    # -- Cursor & epoch grid --
    "cursor": "#FF3B3B",
    "epoch_line": "#585B70",

    # -- Sleep-stage colours (DARK mode – clinical) --
    "stage_W": "#FFD166",
    "stage_N1": "#4DA3FF",
    "stage_N2": "#2F6FE0",
    "stage_N3": "#143A8A",
    "stage_R": "#FF3D7F",
    "stage_U": "#6E6E6E",

    # -- Hypnogram-specific --
    "hypno_bg": "#121826",
    "hypno_grid": "#2A2F3A",
    "hypno_window": "#FFFFFF28",
    "hypno_window_border": "#FFFFFF50",
    "hypno_cycle_line": "#FFFFFF30",
    "hypno_transition_N2N3": "#FFD16640",

    # -- Event colours --
    "event_arousal": "#F9E2AF",
    "event_apnea": "#F38BA8",
    "event_hypopnea": "#FAB387",
    "event_rem": "#CBA6F7",
    "event_default": "#89B4FA",
}

LIGHT: ThemePalette = {
    # -- General UI --
    "background": "#FFFFFF",
    "foreground": "#1E293B",
    "grid": "#E2E8F0",
    "border": "#CBD5E1",
    "surface": "#F8FAFC",
    "surface_alt": "#F1F5F9",
    "accent": "#3B82F6",
    "accent_dim": "#CBD5E1",
    "text_dim": "#94A3B8",

    # -- Signal colours --
    "eeg_color": "#3B82F6",
    "eog_color": "#16A34A",
    "emg_color": "#D97706",
    "ecg_color": "#DC2626",
    "highlight": "#3B82F6",

    # -- Cursor & epoch grid --
    "cursor": "#FF3B3B",
    "epoch_line": "#CBD5E1",

    # -- Sleep-stage colours (LIGHT mode – clinical) --
    "stage_W": "#F4C542",
    "stage_N1": "#6EC6FF",
    "stage_N2": "#3F8CFF",
    "stage_N3": "#1E4DB7",
    "stage_R": "#FF5C93",
    "stage_U": "#B0B0B0",

    # -- Hypnogram-specific --
    "hypno_bg": "#F7F7F7",
    "hypno_grid": "#E0E0E0",
    "hypno_window": "#3B82F620",
    "hypno_window_border": "#3B82F660",
    "hypno_cycle_line": "#00000018",
    "hypno_transition_N2N3": "#F4C54230",

    # -- Event colours --
    "event_arousal": "#D97706",
    "event_apnea": "#DC2626",
    "event_hypopnea": "#EA580C",
    "event_rem": "#7C3AED",
    "event_default": "#3B82F6",
}

THEMES: Dict[str, ThemePalette] = {"dark": DARK, "light": LIGHT}


CHANNEL_TYPE_COLORS: Dict[str, str] = {
    "eeg": "eeg_color",
    "eog": "eog_color",
    "emg": "emg_color",
    "ecg": "ecg_color",
}


_STAGE_ALIASES: Dict[str, str] = {
    "W": "W", "WAKE": "W", "EVEIL": "W", "AWAKE": "W", "0": "W",
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
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def normalize_stage(stage: str) -> str:
    """Map any common stage label to the canonical form (W/N1/N2/N3/R/U)."""
    s = stage.upper().strip().replace(" ", "")
    hit = _STAGE_ALIASES.get(s)
    if hit:
        return hit
    s_ascii = _strip_accents(s)
    hit = _STAGE_ALIASES.get(s_ascii)
    if hit:
        return hit
    for key, val in _STAGE_ALIASES.items():
        if key in s_ascii or s_ascii in key:
            return val
    return "U"


def stage_color(theme: ThemePalette, stage: str) -> str:
    """Return the colour string for a sleep stage label."""
    canonical = normalize_stage(stage)
    key = f"stage_{canonical}"
    return theme.get(key, theme.get("stage_U", "#888888"))


def event_color(theme: ThemePalette, event_type: str) -> str:
    """Return the colour for an event type string."""
    t = event_type.lower()
    if "arousal" in t:
        return theme["event_arousal"]
    if "apnea" in t or "apnée" in t:
        return theme["event_apnea"]
    if "hypopnea" in t or "hypopnée" in t:
        return theme["event_hypopnea"]
    if "rem" in t:
        return theme["event_rem"]
    return theme["event_default"]
