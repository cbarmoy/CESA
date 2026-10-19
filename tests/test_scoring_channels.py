"""Tests for EMG channel-role heuristics used by the scoring backends."""

from __future__ import annotations

import pytest

from CESA.scoring_channels import (
    BACKEND_CHANNEL_PROFILES,
    ROLE_UNUSED,
    EMG_ROLE_LIMB,
    EMG_ROLE_NOT_EMG,
    EMG_ROLE_OTHER_MUSCLE,
    EMG_ROLE_SUBMENTAL,
    EMG_ROLE_UNKNOWN,
    emg_role,
    emg_warning,
    find_limb_emg_candidates,
    find_submental_emg_candidates,
    available_channels,
    channel_profile,
    is_staging_emg,
    non_muscle_modality,
    order_channel_selection,
    profile_candidates,
)


MONTAGE = [
    "C4-M1", "C3-M2", "F4-M1", "E1-M2", "Chin1-Chin2", "Left Leg", "ECG",
]


class TestChannelProfiles:
    def test_usleep_prefers_a_frontal_derivation(self):
        assert channel_profile("usleep").eeg[0] == "F4-M1"
        assert profile_candidates("usleep", "eeg", MONTAGE)[0] == "F4-M1"

    def test_dialog_selection_keeps_usleep_frontal_priority(self):
        # Montage order would put C4-M1 first; the profile must win.
        picked = ["C4-M1", "C3-M2", "F4-M1", "O2-M1"]
        assert order_channel_selection(
            picked, channel_profile("usleep").eeg,
        )[0] == "F4-M1"
        # U-Sleep takes one EEG and one EOG only; an EMG has no effect.
        assert channel_profile("usleep").role("emg") == ROLE_UNUSED
        assert channel_profile("usleep").candidates("emg") == ()

    def test_yasa_recommends_the_emg(self):
        assert channel_profile("yasa").uses("emg")

    def test_yasa_prefers_a_central_derivation(self):
        assert channel_profile("yasa").eeg[0] == "C4-M1"

    def test_pftsleep_expects_referenced_derivations_and_an_ecg(self):
        profile = channel_profile("pftsleep")
        assert profile.uses("ecg")
        assert "F4-M1" not in profile.eeg

    def test_every_backend_documents_its_channel_choice(self):
        for method, profile in BACKEND_CHANNEL_PROFILES.items():
            assert profile.rationale, f"{method} has no rationale"
            assert profile.eeg, f"{method} has no EEG candidate"

    def test_unknown_backend_falls_back_to_a_usable_profile(self):
        assert channel_profile("inconnu").eeg

    def test_profile_candidates_are_restricted_to_the_recording(self):
        assert profile_candidates("yasa", "eeg", MONTAGE) == ["C4-M1", "C3-M2", "F4-M1"]

    def test_profile_candidates_reject_a_limb_emg(self):
        assert profile_candidates("yasa", "emg", ["Left Leg"]) == []

    def test_profile_candidates_are_empty_for_an_unused_modality(self):
        assert profile_candidates("usleep", "emg", MONTAGE) == []

    def test_available_channels_keeps_the_recording_spelling(self):
        assert available_channels(["c4-m1"], ["C4-M1"]) == ["c4-m1"]

    def test_n3_channel_prefers_the_frontal_derivation(self):
        from CESA.scoring_channels import pick_n3_channel

        assert pick_n3_channel(["C4-M1", "F4-M1", "O2-M1"], "C4-M1") == "F4-M1"
        assert pick_n3_channel(["C4-M1"], "C4-M1") == "C4-M1"


@pytest.mark.parametrize("name", [
    "Chin1-Chin2", "EMG Chin", "CHIN", "Menton", "EMG Menton",
    "chin1-chin3", "Submental", "EMG mentonnier",
    "1", "2", "F", "1-2", "1-F", "2-F", "1 2 F", "EMG 1", "EMG F", "EMG2",
])
def test_submental_names_recognised(name):
    assert emg_role(name) == EMG_ROLE_SUBMENTAL
    assert is_staging_emg(name)
    assert emg_warning(name) is None


def test_chin_12f_is_not_confused_with_frontal_eeg():
    assert emg_role("F4") == EMG_ROLE_NOT_EMG
    assert emg_role("F3-M2") == EMG_ROLE_NOT_EMG
    assert emg_role("Fp1") == EMG_ROLE_NOT_EMG


def test_recording_named_12f_is_offered_as_emg():
    names = ["C4-M1", "E1-M2", "1", "2", "F", "Jambe droite"]
    assert find_submental_emg_candidates(names) == ["1", "2", "F"]
    assert profile_candidates("yasa", "emg", names) == ["1", "2", "F"]


def test_french_psg_chin_prefers_bipolar_and_drops_impedance():
    names = [
        "1", "1 Impédance", "1-2", "1-2 (Imp)", "1-F",
        "2", "2 Impédance", "2-F",
    ]
    assert find_submental_emg_candidates(names) == ["1", "1-2", "1-F", "2", "2-F"]
    assert profile_candidates("yasa", "emg", names) == ["1-2", "1-F", "2-F", "1", "2"]


@pytest.mark.parametrize("name", [
    "1 Impédance", "1-2 (Imp)", "2 Impédance", "1 Imp��dance",
])
def test_chin_impedance_traces_are_not_staging_emg(name):
    assert emg_role(name) == EMG_ROLE_NOT_EMG
    assert not is_staging_emg(name)
    assert non_muscle_modality(name) == "voie d'impedance"


@pytest.mark.parametrize("name", [
    "Jambe droite", "Jambe gauche", "EMG Jambe D", "Left Leg", "RIGHT LEG",
    "Tib ant D", "TIBIAL", "PLM", "EMG-Tib-G", "Cheville",
])
def test_limb_names_recognised(name):
    assert emg_role(name) == EMG_ROLE_LIMB
    assert not is_staging_emg(name)


def test_limb_wins_over_generic_emg_prefix():
    # "EMG Jambe D" must not be accepted just because it starts with EMG.
    assert emg_role("EMG Jambe D") == EMG_ROLE_LIMB


def test_bare_emg_stays_unknown_and_allowed():
    assert emg_role("EMG") == EMG_ROLE_UNKNOWN
    assert is_staging_emg("EMG")
    assert emg_warning("EMG") is None


def test_other_muscle_recognised():
    assert emg_role("Masseter") == EMG_ROLE_OTHER_MUSCLE
    assert not is_staging_emg("Masseter")


def test_missing_channel_is_unknown():
    assert emg_role(None) == EMG_ROLE_UNKNOWN
    assert emg_role("") == EMG_ROLE_UNKNOWN
    assert emg_warning(None) is None


def test_limb_warning_explains_rem_atonia_and_alternative():
    message = emg_warning("Jambe droite")
    assert message is not None
    assert "mentonnier" in message
    assert "REM" in message
    assert "mouvements periodiques" in message


def test_other_muscle_warning_mentions_empty_list():
    message = emg_warning("Masseter")
    assert message is not None
    assert "vide" in message


@pytest.mark.parametrize("separator", ["-", "_", " ", "."])
def test_separators_do_not_defeat_matching(separator):
    assert emg_role(f"EMG{separator}Chin") == EMG_ROLE_SUBMENTAL
    assert emg_role(f"EMG{separator}Jambe") == EMG_ROLE_LIMB


@pytest.mark.parametrize("name,modality", [
    ("M1", "reference mastoidienne"),
    ("M2", "reference mastoidienne"),
    ("M1M2", "reference mastoidienne"),
    ("A1", "reference mastoidienne"),
    ("C3", "electrode EEG"),
    # In a derivation the active electrode names the modality, not the reference.
    ("C3-M2", "electrode EEG"),
    ("F4", "electrode EEG"),
    ("O1", "electrode EEG"),
    ("Cz", "electrode EEG"),
    ("E1-M2", "EOG"),
    ("EOG", "EOG"),
    ("ECG", "ECG"),
    ("Fréquence cardi", "ECG"),
    ("Nasal Pressure", "signal respiratoire"),
    ("Thermistance", "signal respiratoire"),
    ("Inductance Thora", "signal respiratoire"),
    ("Abdomen", "signal respiratoire"),
    ("SpO2 B-B", "oxymetrie"),
    ("Saturation", "oxymetrie"),
    ("PosAngle", "capteur de position ou d'activite"),
    ("Activity", "capteur de position ou d'activite"),
    ("X Axis", "capteur de position ou d'activite"),
    ("M1 Impédance", "voie d'impedance"),
])
def test_other_modalities_are_rejected_as_emg(name, modality):
    # Feeding one of these as the EMG input is the worst case: no muscle tone.
    assert emg_role(name) == EMG_ROLE_NOT_EMG
    assert not is_staging_emg(name)
    assert non_muscle_modality(name) == modality


def test_mastoid_reference_warning_names_the_modality():
    message = emg_warning("M1")
    assert message is not None
    assert "n'est pas un EMG" in message
    assert "reference mastoidienne" in message
    assert "Videz la liste EMG" in message


def test_muscle_names_win_over_modality_detection():
    # "EMG Chin1-M2" is still a chin EMG despite the mastoid reference.
    assert emg_role("EMG Chin1-M2") == EMG_ROLE_SUBMENTAL
    assert emg_role("Left Leg-M2") == EMG_ROLE_LIMB
    assert non_muscle_modality("Chin1-Chin2") is None


def test_finders_split_a_realistic_montage():
    montage = [
        "C4-M1", "C3-M2", "E1-M2", "E2-M1", "Chin1-Chin2",
        "Jambe droite", "Jambe gauche", "ECG", "SpO2",
    ]
    assert find_submental_emg_candidates(montage) == ["Chin1-Chin2"]
    assert find_limb_emg_candidates(montage) == ["Jambe droite", "Jambe gauche"]
