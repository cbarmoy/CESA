"""Channel roles and per-backend channel profiles for automatic sleep scoring.

Two concerns live here, both about "which channel goes where".

First, channel-role heuristics. Staging models expect a *submental* (chin) EMG:
REM is defined by muscle atonia of the chin. A limb EMG (tibialis anterior,
used for periodic limb movements) carries a different signal entirely, and a
mastoid or an EOG carries no muscle tone at all; feeding any of them as the EMG
input degrades REM detection instead of helping.

Second, the channel priorities each backend actually expects, which differ:
U-Sleep ignores the EMG altogether, PFTSleep wants referenced derivations plus
an ECG, and the rule-based backends benefit from a frontal derivation for slow
waves. Keeping one profile per backend here stops the UI from offering the same
YASA-shaped list for every algorithm.

Kept free of MNE and Qt imports so it can be used from anywhere.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: Substrings identifying a submental / chin EMG, the only EMG a staging model
#: can exploit.
SUBMENTAL_TOKENS: tuple[str, ...] = (
    "CHIN", "MENTON", "MENTONNIER", "SUBMENTAL", "SUBMENT", "KINN",
)

#: AASM three-electrode chin montage, often exported under these short names
#: (left, right, midline). Matched only as whole channel names, never as a
#: letter inside ``F4`` or a digit inside ``C3``.
_CHIN_12F_EXACT: frozenset[str] = frozenset({"1", "2", "F"})
_CHIN_12F_RE = re.compile(
    r"^(?:EMG\s+)?(?:1\s+2(?:\s+F)?|1\s+F|2\s+F)$"
)
_CHIN_12F_EMG_RE = re.compile(r"^EMG\s*[12F]$")

#: Substrings identifying a limb EMG, recorded for periodic limb movements.
LIMB_TOKENS: tuple[str, ...] = (
    "JAMBE", "LEG", "TIB", "TIBIAL", "PLM", "PLMS", "ANKLE", "CHEVILLE",
    "MOLLET", "CALF", "ARM", "BRAS", "BEIN",
)

#: Other muscles occasionally recorded, equally unusable for staging.
OTHER_MUSCLE_TOKENS: tuple[str, ...] = (
    "MASSETER", "INTERCOST", "DIAPHRAG", "STERNO",
)

#: Channels belonging to another modality entirely. Selecting one of these as
#: the EMG input feeds the model a signal that carries no muscle tone at all,
#: which is worse than leaving the EMG unset.
#: Order matters: in a derivation such as ``E1-M2`` the *active* electrode names
#: the modality, so the mastoid reference is only consulted last.
NON_MUSCLE_TOKENS: dict[str, tuple[str, ...]] = {
    "EOG": ("EOG", "LOC", "ROC", "E1", "E2", "EYE", "OEIL"),
    "ECG": ("ECG", "EKG", "CARDI", "PULSE", "HEART", "PTT", "PWA"),
    "signal respiratoire": (
        "FLOW", "AIRFLOW", "THOR", "ABDO", "ABDOMEN", "CHEST", "RIP",
        "NASAL", "SNOR", "RONFL", "THERMIST", "CANNULE", "PRESSION",
        "RESP", "INDUCTANCE",
    ),
    "oxymetrie": ("SPO2", "SAO2", "SATUR", "OXY"),
    "capteur de position ou d'activite": (
        "POS", "POSANGLE", "ACTIVITY", "ELEVATION", "AXIS", "LIGHT",
        "AUDIO", "VOLUME", "TENSION", "BATTERY", "PILE",
    ),
    "voie d'impedance": ("IMP", "IMPEDANCE", "IMPED",),
    "electrode EEG": ("CZ", "FZ", "PZ", "OZ", "FPZ", "FCZ", "CPZ", "POZ", "EEG"),
}

#: Consulted only after every active electrode type, so ``C3-M2`` is reported as
#: an EEG derivation rather than as a mastoid.
MASTOID_TOKENS: tuple[str, ...] = (
    "M1", "M2", "M1M2", "A1", "A2", "A1A2", "MASTOID",
)

#: 10-20 electrode names such as C3, F4, O1, T5, FC2: an EEG derivation used as
#: the EMG input.
_EEG_ELECTRODE_RE = re.compile(
    r"^(FP|AF|F|FC|C|CP|P|PO|O|T|TP|FT)\d{1,2}$"
)

EMG_ROLE_SUBMENTAL = "submental"
EMG_ROLE_LIMB = "limb"
EMG_ROLE_OTHER_MUSCLE = "other_muscle"
EMG_ROLE_NOT_EMG = "not_emg"
EMG_ROLE_UNKNOWN = "unknown"


def _normalised_channel_name(name: str) -> str:
    stripped = unicodedata.normalize("NFKD", str(name))
    stripped = "".join(ch for ch in stripped if not unicodedata.combining(ch))
    return re.sub(r"[^A-Z0-9]+", " ", stripped.upper()).strip()


def _is_chin_12f(name: Optional[str]) -> bool:
    """True for the short 1 / 2 / F chin labels used on many French PSGs."""
    if not name:
        return False
    compact = _normalised_channel_name(name)
    return (
        compact in _CHIN_12F_EXACT
        or bool(_CHIN_12F_EMG_RE.fullmatch(compact))
        or bool(_CHIN_12F_RE.fullmatch(compact))
    )


def _tokens(name: str) -> str:
    """Uppercase name with separators normalised to spaces for word matching."""
    return " " + _normalised_channel_name(name) + " "


def _contains(haystack: str, tokens: Iterable[str]) -> bool:
    """Substring match, so CHIN also matches CHIN1 and TIB matches TIBIAL."""
    return any(token in haystack for token in tokens)


def _starts_word(haystack: str, tokens: Iterable[str]) -> bool:
    """Word-prefix match, to keep short codes such as M1 or E1 unambiguous."""
    words = haystack.split()
    return any(word.startswith(token) for word in words for token in tokens)


def non_muscle_modality(name: Optional[str]) -> Optional[str]:
    """Modality of *name* when it is clearly not a muscle channel.

    Returns a human-readable modality label, or None when the name could
    plausibly be an EMG.
    """
    if not name:
        return None
    text = _tokens(name)
    if (
        _contains(text, SUBMENTAL_TOKENS)
        or _contains(text, LIMB_TOKENS)
        or _is_chin_12f(name)
    ):
        return None
    for modality, tokens in NON_MUSCLE_TOKENS.items():
        if _starts_word(text, tokens):
            return modality
    for word in text.split():
        if _EEG_ELECTRODE_RE.match(word):
            return "electrode EEG"
    if _starts_word(text, MASTOID_TOKENS):
        return "reference mastoidienne"
    return None


def emg_role(name: Optional[str]) -> str:
    """Classify an EMG channel name by the muscle it most likely records.

    Returns one of ``submental``, ``limb``, ``other_muscle``, ``not_emg`` or
    ``unknown``. A bare ``EMG`` stays ``unknown``: by convention it is usually
    the chin, but nothing in the name proves it.
    """
    if not name:
        return EMG_ROLE_UNKNOWN
    text = _tokens(name)
    # Impedance traces of the chin electrodes must not be scored as EMG,
    # even when the rest of the name is 1-2 / 1-F.
    if _starts_word(text, NON_MUSCLE_TOKENS["voie d'impedance"]):
        return EMG_ROLE_NOT_EMG
    # Limb wins over a generic "EMG" prefix: "EMG Jambe D" is a limb channel.
    if _contains(text, LIMB_TOKENS):
        return EMG_ROLE_LIMB
    if _contains(text, SUBMENTAL_TOKENS) or _is_chin_12f(name):
        return EMG_ROLE_SUBMENTAL
    if _contains(text, OTHER_MUSCLE_TOKENS):
        return EMG_ROLE_OTHER_MUSCLE
    if non_muscle_modality(name) is not None:
        return EMG_ROLE_NOT_EMG
    return EMG_ROLE_UNKNOWN


def is_staging_emg(name: Optional[str]) -> bool:
    """True when the channel can serve as the EMG input of a staging model."""
    return emg_role(name) in (EMG_ROLE_SUBMENTAL, EMG_ROLE_UNKNOWN)


def emg_warning(name: Optional[str]) -> Optional[str]:
    """Message explaining why an EMG channel is unfit for staging, if it is."""
    role = emg_role(name)
    if role == EMG_ROLE_LIMB:
        return (
            f"Le canal EMG '{name}' est un EMG de membre (jambe / tibial). Les "
            "modeles de scoring attendent un EMG mentonnier : le REM se definit "
            "par l'atonie du menton, que l'EMG de jambe ne mesure pas. Fourni "
            "comme EMG, il degrade la detection du REM au lieu de l'ameliorer. "
            "Retirez-le de la liste EMG : sans EMG, le modele se rabat sur "
            "l'EEG (et l'EOG), ce qui est preferable. Gardez ce canal pour la "
            "detection des mouvements periodiques des jambes."
        )
    if role == EMG_ROLE_OTHER_MUSCLE:
        return (
            f"Le canal EMG '{name}' ne semble pas etre un EMG mentonnier. Les "
            "modeles de scoring attendent le menton (atonie du REM) ; un autre "
            "muscle degrade la detection du REM. Laissez la liste EMG vide si "
            "aucun canal mentonnier n'est disponible."
        )
    if role == EMG_ROLE_NOT_EMG:
        modality = non_muscle_modality(name) or "autre modalite"
        return (
            f"Le canal '{name}' fourni comme EMG n'est pas un EMG : c'est une "
            f"{modality}. Il ne contient aucun tonus musculaire, donc le modele "
            "recoit une entree denuee de sens a la place de l'atonie du menton "
            "qu'il attend. Videz la liste EMG : le modele se rabattra sur l'EEG "
            "et l'EOG, ce qui donne un meilleur resultat qu'un EMG factice."
        )
    return None


def find_limb_emg_candidates(names: Sequence[str]) -> List[str]:
    """Limb EMG channels among the given names."""
    return [name for name in names if emg_role(name) == EMG_ROLE_LIMB]


ROLE_REQUIRED = "requis"
ROLE_RECOMMENDED = "recommande"
ROLE_OPTIONAL = "optionnel"
ROLE_UNUSED = "non utilise"


@dataclass(frozen=True)
class ChannelProfile:
    """What one backend expects, in its own order of preference."""

    eeg: Tuple[str, ...]
    eog: Tuple[str, ...] = ()
    emg: Tuple[str, ...] = ()
    ecg: Tuple[str, ...] = ()
    eeg_role: str = ROLE_REQUIRED
    eog_role: str = ROLE_RECOMMENDED
    emg_role: str = ROLE_RECOMMENDED
    ecg_role: str = ROLE_UNUSED
    rationale: str = ""

    def role(self, modality: str) -> str:
        return {
            "eeg": self.eeg_role,
            "eog": self.eog_role,
            "emg": self.emg_role,
            "ecg": self.ecg_role,
        }.get(modality, ROLE_OPTIONAL)

    def candidates(self, modality: str) -> Tuple[str, ...]:
        if self.role(modality) == ROLE_UNUSED:
            return ()
        return {
            "eeg": self.eeg, "eog": self.eog, "emg": self.emg, "ecg": self.ecg,
        }.get(modality, ())

    def uses(self, modality: str) -> bool:
        return self.role(modality) != ROLE_UNUSED


#: Central derivations, the reference for every staging model.
_CENTRAL_EEG: Tuple[str, ...] = (
    "C4-M1", "C4-A1", "C3-M2", "C3-A2", "C4", "C3",
)
_FRONTAL_EEG: Tuple[str, ...] = ("F4-M1", "F4-A1", "F3-M2", "F3-A2", "F4", "F3")
_OCCIPITAL_EEG: Tuple[str, ...] = ("O2-M1", "O2-A1", "O1-M2", "O1-A2", "O2", "O1")

_STANDARD_EOG: Tuple[str, ...] = (
    "E1-M2", "E2-M1", "E1-A2", "E2-A1", "EOG", "LOC", "ROC",
    "EOG gauche", "EOG droite", "EOG Left", "EOG Right",
)
_SUBMENTAL_EMG: Tuple[str, ...] = (
    "1-2", "1-F", "2-F", "1-2 F",
    "Chin1-Chin2", "Chin1-Chin3", "EMG Chin", "EMG Menton", "CHIN", "MENTON",
    "EMG 1", "EMG 2", "EMG F", "1", "2", "F", "EMG",
)
_ECG: Tuple[str, ...] = ("ECG", "ECG1", "ECG2", "EKG")

#: One profile per backend. Priorities follow what each model was trained on,
#: which is why they are not interchangeable.
BACKEND_CHANNEL_PROFILES: Dict[str, ChannelProfile] = {
    "yasa": ChannelProfile(
        eeg=_CENTRAL_EEG + _FRONTAL_EEG + _OCCIPITAL_EEG,
        eog=_STANDARD_EOG,
        emg=_SUBMENTAL_EMG,
        rationale=(
            "YASA a ete entraine sur une derivation centrale (C4-M1), un EOG et "
            "un EMG mentonnier. La centrale est donc a privilegier ; la frontale "
            "et l'occipitale fonctionnent mais degradent la detection des fuseaux "
            "et du sommeil lent."
        ),
    ),
    "usleep": ChannelProfile(
        eeg=_FRONTAL_EEG + _CENTRAL_EEG + _OCCIPITAL_EEG,
        eog=_STANDARD_EOG,
        emg=(),
        emg_role=ROLE_UNUSED,
        rationale=(
            "U-Sleep est agnostique au montage (1 EEG + 1 EOG, pas d'EMG). La "
            "frontale (F4-M1) est proposee en tete : c'est le canal du bandeau "
            "et celui ou le N3 se lit le mieux ; la centrale reste un repli."
        ),
    ),
    "pftsleep": ChannelProfile(
        eeg=("C4-M1", "C3-M2", "FPZ-CZ", "PZ-OZ"),
        eog=("E1-M2", "E2-M1", "EOG L", "EOG LEFT", "EOG GAUCHE"),
        emg=("1-2", "Chin1-Chin2", "Chin1-Chin3", "EMG Chin", "CHIN", "MENTON",
             "1", "2", "F"),
        ecg=_ECG,
        ecg_role=ROLE_RECOMMENDED,
        rationale=(
            "PFTSleep attend des derivations referencees et exploite en plus "
            "l'ECG, la SpO2 et les sangles respiratoires. Les canaux absents sont "
            "remplaces par un vecteur nul, ce qui degrade la prediction : ce "
            "backend n'a d'interet que sur un montage PSG complet."
        ),
    ),
    "aasm_rules": ChannelProfile(
        eeg=_CENTRAL_EEG + _FRONTAL_EEG + _OCCIPITAL_EEG,
        eog=_STANDARD_EOG,
        emg=_SUBMENTAL_EMG,
        rationale=(
            "Les regles AASM s'appliquent aux trois derivations recommandees "
            "(F4-M1 pour les ondes lentes, C4-M1 pour les fuseaux et complexes K, "
            "O2-M1 pour l'alpha). L'implementation n'en lit qu'une : la centrale "
            "est le meilleur compromis, la frontale si le sommeil lent est "
            "l'objet de l'etude."
        ),
    ),
    "rules_hmm": ChannelProfile(
        eeg=_CENTRAL_EEG + _FRONTAL_EEG + _OCCIPITAL_EEG,
        eog=_STANDARD_EOG,
        emg=_SUBMENTAL_EMG,
        rationale=(
            "Memes canaux que les regles AASM ; le decodage HMM ne porte que sur "
            "la sequence de stades, pas sur le signal."
        ),
    ),
    "ml": ChannelProfile(
        eeg=_CENTRAL_EEG + _FRONTAL_EEG + _OCCIPITAL_EEG,
        eog=_STANDARD_EOG,
        emg=_SUBMENTAL_EMG,
        rationale=(
            "Le montage doit reproduire celui de l'entrainement de votre modele : "
            "changer de derivation invalide les descripteurs appris."
        ),
    ),
    "ml_hmm": ChannelProfile(
        eeg=_CENTRAL_EEG + _FRONTAL_EEG + _OCCIPITAL_EEG,
        eog=_STANDARD_EOG,
        emg=_SUBMENTAL_EMG,
        rationale=(
            "Memes canaux que le backend ML : le montage doit rester celui de "
            "l'entrainement."
        ),
    ),
}

#: Fallback for an unknown backend name.
DEFAULT_CHANNEL_PROFILE = BACKEND_CHANNEL_PROFILES["yasa"]


def channel_profile(method: Optional[str]) -> ChannelProfile:
    """Channel expectations of *method*, falling back to the YASA-like profile."""
    key = str(method or "").lower().strip()
    return BACKEND_CHANNEL_PROFILES.get(key, DEFAULT_CHANNEL_PROFILE)


def available_channels(
    names: Sequence[str], candidates: Sequence[str],
) -> List[str]:
    """Candidates present in *names*, in candidate order, original spelling kept."""
    by_upper = {str(n).upper().strip(): str(n) for n in names}
    out: List[str] = []
    for candidate in candidates:
        hit = by_upper.get(str(candidate).upper().strip())
        if hit is not None and hit not in out:
            out.append(hit)
    return out


def order_channel_selection(
    picked: Sequence[str], priority: Sequence[str],
) -> List[str]:
    """Keep the user's ticks, but honour the backend's candidate order.

    The configuration dialog lists channels in montage order. Without this,
    confirming the dialog would put ``C4-M1`` first even when the backend
    prefers ``F4-M1``.
    """
    if not priority:
        return list(picked)
    ordered = available_channels(picked, priority)
    seen = {str(name).upper().strip() for name in ordered}
    for name in picked:
        key = str(name).upper().strip()
        if key not in seen:
            ordered.append(name)
            seen.add(key)
    return ordered


def profile_candidates(
    method: Optional[str], modality: str, names: Optional[Sequence[str]] = None,
) -> List[str]:
    """Channels *method* would consider for *modality*.

    With *names*, restricted to what the recording actually contains. EMG
    candidates are additionally filtered to those usable for staging, so a limb
    channel is never proposed.
    """
    profile = channel_profile(method)
    if modality == "emg" and not profile.uses("emg"):
        return []
    candidates = list(profile.candidates(modality))
    if names is not None:
        candidates = available_channels(names, candidates)
        if modality == "emg":
            for extra in find_submental_emg_candidates(names):
                if extra not in candidates:
                    candidates.append(extra)
    if modality == "emg":
        candidates = [name for name in candidates if is_staging_emg(name)]
    return candidates


_FRONTAL_N3_PRIORITY: Tuple[str, ...] = (
    "F4-M1", "F4-A1", "F3-M2", "F3-A2", "F4", "F3",
)


def pick_n3_channel(
    names: Sequence[str], fallback: Optional[str] = None,
) -> Optional[str]:
    """Frontal derivation for the AASM N3 criterion, else *fallback*.

    Slow waves are maximal frontally; measuring the criterion on C4 under-calls
    N3 compared to F4-M1, which is the AASM recommended derivation.
    """
    found = available_channels(names, _FRONTAL_N3_PRIORITY)
    if found:
        return found[0]
    return fallback


def find_submental_emg_candidates(names: Sequence[str]) -> List[str]:
    """Submental EMG channels among the given names."""
    return [name for name in names if emg_role(name) == EMG_ROLE_SUBMENTAL]
