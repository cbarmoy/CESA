"""One-click multi-filter display recipes for a single physical channel.

Literature / clinical practice often scores sleep while viewing the same
derivation under several bandpasses at once, e.g.:

* Visbrain *Sleep* (Combrisson et al., 2017): toggle spindle band (12–14 Hz)
  over the clinical trace for N2 marking.
* AASM / PSG scoring: wide-band EEG (≈0.3–35 Hz) for stage morphology, with
  optional sigma emphasis for spindles and delta for slow waves / K-complexes.
* Spindle typology: slow (≈10–13 Hz, frontal) vs fast (≈13–16 Hz, centro-parietal).
* SO–spindle coupling work: simultaneous delta/SO (0.5–4 Hz) and sigma (11–16 Hz).
* Wake vs N1: alpha (8–12 Hz) and/or beta (15–30 Hz) beside the standard band.
* REM: EOG wide vs REM-oriented band (≈0.5–5 Hz) for rapid eye movements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from CESA.filter_engine import (
    BandpassFilter,
    FilterPipeline,
    NotchFilter,
)


@dataclass(frozen=True)
class MultiViewLane:
    """One display strip in a multi-view recipe."""

    short_name: str
    """Suffix shown in the label, e.g. ``Spindles`` → ``C3 · Spindles``."""

    color: str
    """Hex colour for the trace."""

    pipeline: FilterPipeline
    """Independent filter chain for this strip."""

    is_primary: bool = False
    """If True, configures the original channel (not a clone)."""


@dataclass(frozen=True)
class MultiViewRecipe:
    """Named bundle of lanes applied to one physical source channel."""

    recipe_id: str
    title: str
    description: str
    channel_types: tuple  # e.g. ("eeg",) or ("eog",)
    lanes: tuple  # Tuple[MultiViewLane, ...]


def _bp(lo: float, hi: float, order: int = 4) -> FilterPipeline:
    return FilterPipeline(
        filters=[
            BandpassFilter(
                low_hz=lo, high_hz=hi, order=order,
                filter_type="butterworth", causal=False,
            ),
        ],
        enabled=True,
    )


def _bp_notch(lo: float, hi: float, notch_hz: float = 50.0) -> FilterPipeline:
    return FilterPipeline(
        filters=[
            BandpassFilter(
                low_hz=lo, high_hz=hi, order=4,
                filter_type="butterworth", causal=False,
            ),
            NotchFilter(freq_hz=notch_hz, quality_factor=30.0, harmonics=1),
        ],
        enabled=True,
    )


def _raw() -> FilterPipeline:
    return FilterPipeline(filters=[], enabled=False)


def _standard_psg() -> FilterPipeline:
    return _bp_notch(0.3, 35.0)


# Distinct colours (Catppuccin-ish) aligned with event colours in themes.py
_C_STD = "#89B4FA"       # clinical EEG blue
_C_SPINDLE = "#CBA6F7"   # violet — spindles
_C_DELTA = "#74C7EC"     # cyan — slow waves
_C_K = "#A6E3A1"         # green — K-complex
_C_ALPHA = "#F9E2AF"     # yellow — alpha / wake
_C_BETA = "#FAB387"      # peach — beta / arousal
_C_RAW = "#A6ADC8"       # grey — raw
_C_SLOW_SP = "#B4BEFE"   # lavender — slow spindles
_C_FAST_SP = "#F5C2E7"   # pink — fast spindles
_C_EOG = "#94E2D5"       # teal — EOG standard
_C_EOG_REM = "#CBA6F7"   # violet — REM bursts


def build_builtin_recipes() -> List[MultiViewRecipe]:
    """Return the built-in multi-view recipes (literature-oriented)."""
    return [
        MultiViewRecipe(
            recipe_id="n2_scoring",
            title="N2 · Standard + Spindles + K-complexe",
            description=(
                "Vue classique pour scorer le N2 : bande clinique (0.3–35 Hz), "
                "sigma spindles (11–16 Hz) et bande K-complexe (0.5–5 Hz). "
                "Inspiree de la pratique PSG / Visbrain Sleep."
            ),
            channel_types=("eeg",),
            lanes=(
                MultiViewLane("Standard", _C_STD, _standard_psg(), is_primary=True),
                MultiViewLane("Spindles", _C_SPINDLE, _bp(11.0, 16.0)),
                MultiViewLane("K-complexe", _C_K, _bp(0.5, 5.0)),
            ),
        ),
        MultiViewRecipe(
            recipe_id="n3_delta",
            title="N3 · Standard + Delta/SO",
            description=(
                "Sommeil profond : trace clinique + ondes lentes / SO (0.5–4 Hz). "
                "Utile pour confirmer le N3 et les slow waves."
            ),
            channel_types=("eeg",),
            lanes=(
                MultiViewLane("Standard", _C_STD, _standard_psg(), is_primary=True),
                MultiViewLane("Delta/SO", _C_DELTA, _bp(0.5, 4.0)),
            ),
        ),
        MultiViewRecipe(
            recipe_id="so_spindle",
            title="Couplage SO–Spindle · Delta + Sigma",
            description=(
                "Deux bandes utilisees ensemble dans les etudes de couplage "
                "slow-oscillation / spindle : 0.5–4 Hz et 11–16 Hz."
            ),
            channel_types=("eeg",),
            lanes=(
                MultiViewLane("Delta/SO", _C_DELTA, _bp(0.5, 4.0), is_primary=True),
                MultiViewLane("Spindles", _C_SPINDLE, _bp(11.0, 16.0)),
            ),
        ),
        MultiViewRecipe(
            recipe_id="spindle_types",
            title="Spindles · Lents + Rapides (+ Standard)",
            description=(
                "Typologie spindles : lents ~10–13 Hz (plutot frontaux) vs "
                "rapides ~13–16 Hz (centro-parietaux), avec la vue clinique."
            ),
            channel_types=("eeg",),
            lanes=(
                MultiViewLane("Standard", _C_STD, _standard_psg(), is_primary=True),
                MultiViewLane("Sp. lents", _C_SLOW_SP, _bp(10.0, 13.0)),
                MultiViewLane("Sp. rapides", _C_FAST_SP, _bp(13.0, 16.0)),
            ),
        ),
        MultiViewRecipe(
            recipe_id="wake_n1",
            title="Eveil / N1 · Standard + Alpha + Beta",
            description=(
                "Discriminer eveil et N1 : bande clinique, alpha (8–12 Hz) "
                "et beta (15–30 Hz) pour arousals / intrusion alpha."
            ),
            channel_types=("eeg",),
            lanes=(
                MultiViewLane("Standard", _C_STD, _standard_psg(), is_primary=True),
                MultiViewLane("Alpha", _C_ALPHA, _bp(8.0, 12.0)),
                MultiViewLane("Beta", _C_BETA, _bp(15.0, 30.0)),
            ),
        ),
        MultiViewRecipe(
            recipe_id="raw_vs_filtered",
            title="Brut vs Filtre · Raw + Standard PSG",
            description=(
                "Controle qualite : signal brut a cote du filtrage clinique "
                "0.3–35 Hz + notch 50 Hz."
            ),
            channel_types=("eeg", "eog", "emg", "ecg", "generic"),
            lanes=(
                MultiViewLane("Brut", _C_RAW, _raw(), is_primary=True),
                MultiViewLane("Standard", _C_STD, _standard_psg()),
            ),
        ),
        MultiViewRecipe(
            recipe_id="eog_rem",
            title="EOG · Standard + REM",
            description=(
                "Mouvements oculaires : EOG large (0.3–10 Hz) et bande REM "
                "(0.5–5 Hz) pour les bursts / REM."
            ),
            channel_types=("eog",),
            lanes=(
                MultiViewLane(
                    "EOG std",
                    _C_EOG,
                    _bp(0.3, 10.0, order=2),
                    is_primary=True,
                ),
                MultiViewLane("EOG REM", _C_EOG_REM, _bp(0.5, 5.0, order=2)),
            ),
        ),
        MultiViewRecipe(
            recipe_id="full_night_bands",
            title="Panorama · Standard + Delta + Spindles + Alpha",
            description=(
                "Quatre vues pour un scoring polyvalent : clinique, delta, "
                "spindles et alpha (occupe plus d'espace vertical)."
            ),
            channel_types=("eeg",),
            lanes=(
                MultiViewLane("Standard", _C_STD, _standard_psg(), is_primary=True),
                MultiViewLane("Delta/SO", _C_DELTA, _bp(0.5, 4.0)),
                MultiViewLane("Spindles", _C_SPINDLE, _bp(11.0, 16.0)),
                MultiViewLane("Alpha", _C_ALPHA, _bp(8.0, 12.0)),
            ),
        ),
    ]


_RECIPES: Optional[Dict[str, MultiViewRecipe]] = None


def get_recipes() -> List[MultiViewRecipe]:
    return list(build_builtin_recipes())


def get_recipe(recipe_id: str) -> Optional[MultiViewRecipe]:
    global _RECIPES
    if _RECIPES is None:
        _RECIPES = {r.recipe_id: r for r in build_builtin_recipes()}
    return _RECIPES.get(recipe_id)


def recipes_for_channel_type(channel_type: str) -> List[MultiViewRecipe]:
    ct = (channel_type or "eeg").strip().lower()
    out: List[MultiViewRecipe] = []
    for r in get_recipes():
        types = tuple(t.lower() for t in r.channel_types)
        if ct in types or "generic" in types:
            out.append(r)
    return out


def make_lane_label(source_name: str, short_name: str) -> str:
    """``C3-M2 · Spindles`` style label."""
    src = (source_name or "").strip() or "Canal"
    short = (short_name or "").strip() or "Vue"
    return f"{src} · {short}"
