"""Dark / light theme palettes for the PyQtGraph viewer."""

from __future__ import annotations

from typing import Dict

ThemePalette = Dict[str, str]

DARK: ThemePalette = {
    "background": "#1E1E2E",
    "foreground": "#CDD6F4",
    "grid": "#45475A",
    "border": "#585B70",
    "surface": "#313244",
    "surface_alt": "#2A2A3C",
    "accent": "#89B4FA",
    "accent_dim": "#45475A",
    "eeg_color": "#89B4FA",
    "eog_color": "#A6E3A1",
    "emg_color": "#F9E2AF",
    "ecg_color": "#F38BA8",
    "highlight": "#89B4FA",
    "cursor": "#CBA6F7",
    "epoch_line": "#585B70",
    "text_dim": "#6C7086",
    # Sleep-stage colors (AASM conventional)
    "stage_W": "#1f77b4",
    "stage_N1": "#ff7f0e",
    "stage_N2": "#2ca02c",
    "stage_N3": "#d62728",
    "stage_R": "#9467bd",
    "stage_U": "#6C7086",
    # Event colors
    "event_arousal": "#F9E2AF",
    "event_apnea": "#F38BA8",
    "event_hypopnea": "#FAB387",
    "event_rem": "#CBA6F7",
    "event_default": "#89B4FA",
}

LIGHT: ThemePalette = {
    "background": "#FFFFFF",
    "foreground": "#1E293B",
    "grid": "#E2E8F0",
    "border": "#CBD5E1",
    "surface": "#F8FAFC",
    "surface_alt": "#F1F5F9",
    "accent": "#3B82F6",
    "accent_dim": "#CBD5E1",
    "eeg_color": "#3B82F6",
    "eog_color": "#16A34A",
    "emg_color": "#D97706",
    "ecg_color": "#DC2626",
    "highlight": "#3B82F6",
    "cursor": "#7C3AED",
    "epoch_line": "#CBD5E1",
    "text_dim": "#94A3B8",
    "stage_W": "#1f77b4",
    "stage_N1": "#ff7f0e",
    "stage_N2": "#2ca02c",
    "stage_N3": "#d62728",
    "stage_R": "#9467bd",
    "stage_U": "#94A3B8",
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


def stage_color(theme: ThemePalette, stage: str) -> str:
    """Return the colour string for a sleep stage label."""
    key = f"stage_{stage.upper().strip()}"
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
