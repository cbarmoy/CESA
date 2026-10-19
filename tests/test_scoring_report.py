"""Tests for the enriched auto-vs-manual comparison report."""

from __future__ import annotations

import pandas as pd
import pytest

from CESA.scoring_report import (
    agreement_summary,
    alignment_rows,
    interpretation_notes,
    merged_deep_sleep_accuracy,
    params_rows,
    top_confusions,
    write_report_csv,
)


def _merged(auto, manual):
    auto = list(auto)
    manual = list(manual)
    return pd.DataFrame({
        "epoch": range(len(auto)),
        "auto": auto,
        "manual": manual,
        "match": [a == m for a, m in zip(auto, manual)],
    })


def _diag(**overrides):
    diag = {
        "auto_method": "yasa",
        "manual_source": "edf_embarque",
        "n_auto": 10,
        "n_manual": 10,
        "auto_epoch_seconds": 30.0,
        "manual_epoch_seconds": 30.0,
        "comparison_epoch_seconds": 30.0,
        "n_common": 10,
        "n_compared": 10,
        "n_excluded_u": 0,
        "n_excluded_night": 0,
        "exclude_unscored": True,
        "never_predicted": [],
        "auto_params": {
            "backend": "yasa",
            "canal_eeg": "C4-A1",
            "canal_emg": None,
            "epoque_s": 30.0,
            "seuil_confiance": 0.5,
            "repartition_stades": {"N2": {"n": 6, "pct": 60.0}, "W": {"n": 4, "pct": 40.0}},
        },
    }
    diag.update(overrides)
    return diag


def test_agreement_summary_counts_matches():
    merged = _merged(["W", "W", "N2", "N2"], ["W", "N1", "N2", "N2"])
    summary = agreement_summary(merged)
    assert summary["n_total"] == 4
    assert summary["n_match"] == 3
    assert summary["n_mismatch"] == 1
    assert summary["accuracy"] == pytest.approx(75.0)


def test_agreement_summary_handles_empty_frame():
    merged = _merged([], [])
    summary = agreement_summary(merged)
    assert summary["n_total"] == 0
    assert summary["accuracy"] == 0.0


def test_top_confusions_ranked_by_frequency():
    merged = _merged(
        ["N2", "N2", "N2", "W", "R"],
        ["N3", "N3", "N3", "R", "R"],
    )
    confusions = top_confusions(merged)
    assert confusions[0]["manual"] == "N3"
    assert confusions[0]["auto"] == "N2"
    assert confusions[0]["n"] == 3
    assert confusions[0]["pct_errors"] == pytest.approx(75.0)
    assert confusions[0]["pct_total"] == pytest.approx(60.0)


def test_top_confusions_empty_when_perfect():
    assert top_confusions(_merged(["W", "N2"], ["W", "N2"])) == []


def test_merged_deep_sleep_accuracy_absorbs_n2_n3_boundary():
    merged = _merged(["N2", "N2", "N2", "W"], ["N3", "N3", "N2", "W"])
    assert merged_deep_sleep_accuracy(merged) == pytest.approx(100.0)
    assert agreement_summary(merged)["accuracy"] == pytest.approx(50.0)


def test_interpretation_flags_never_predicted_stage():
    merged = _merged(["N2"] * 8 + ["W", "W"], ["N3"] * 8 + ["W", "W"])
    notes = interpretation_notes(merged, _diag(never_predicted=["N3"]))
    joined = " ".join(notes)
    assert "jamais predit" in joined
    assert "N3" in joined
    assert "amplitude" in joined


def test_interpretation_flags_accuracy_below_backend_expectation():
    merged = _merged(["N2"] * 5 + ["W"] * 5, ["N3"] * 5 + ["W"] * 5)
    notes = interpretation_notes(merged, _diag())
    assert any("inferieure a l'attendu" in note for note in notes)


def test_interpretation_confirms_expected_accuracy():
    merged = _merged(["N2"] * 9 + ["W"], ["N2"] * 9 + ["R"])
    notes = interpretation_notes(merged, _diag())
    assert any("conforme a l'attendu" in note for note in notes)


def test_interpretation_suggests_filling_yasa_metadata():
    merged = _merged(["N2"] * 5 + ["W"] * 5, ["N3"] * 5 + ["W"] * 5)
    diag = _diag()
    diag["auto_params"].update({"age": None, "sexe": None})
    notes = " ".join(interpretation_notes(merged, diag))
    assert "metadonnees" in notes


def test_metadata_suggestion_omitted_when_age_is_known():
    merged = _merged(["N2"] * 5 + ["W"] * 5, ["N3"] * 5 + ["W"] * 5)
    diag = _diag()
    diag["auto_params"].update({"age": 34, "sexe": "homme"})
    assert "metadonnees" not in " ".join(interpretation_notes(merged, diag))


def test_metadata_suggestion_omitted_when_result_is_already_good():
    merged = _merged(["N2"] * 9 + ["W"], ["N2"] * 9 + ["R"])
    diag = _diag()
    diag["auto_params"].update({"age": None, "sexe": None})
    assert "metadonnees" not in " ".join(interpretation_notes(merged, diag))


def test_interpretation_always_reports_kappa():
    notes = interpretation_notes(_merged(["W", "N2"], ["W", "N2"]), _diag())
    assert any("Kappa de Cohen" in note for note in notes)


def test_interpretation_handles_empty_comparison():
    notes = interpretation_notes(_merged([], []), _diag())
    assert notes == ["Aucune epoque comparable."]


def test_interpretation_without_diagnostics_does_not_crash():
    notes = interpretation_notes(_merged(["W", "N2"], ["W", "R"]), None)
    assert notes


def test_interpretation_flags_limb_emg_used_as_staging_input():
    merged = _merged(["W"] * 25 + ["N2"] * 5, ["R"] * 25 + ["N2"] * 5)
    diag = _diag()
    diag["auto_params"]["canal_emg"] = "Jambe droite"
    notes = " ".join(interpretation_notes(merged, diag))
    assert "EMG de membre" in notes
    assert "mentonnier" in notes


def test_interpretation_rem_as_wake_note_does_not_recommend_leg_emg():
    merged = _merged(["W"] * 25 + ["N2"] * 5, ["R"] * 25 + ["N2"] * 5)
    diag = _diag()
    diag["auto_params"]["canal_emg"] = None
    notes = [n for n in interpretation_notes(merged, diag) if "classees eveil" in n]
    assert notes, "the REM-as-wake note should be emitted"
    assert "un EMG de jambe, non" in notes[0]


def test_interpretation_silent_on_emg_when_chin_channel_used():
    merged = _merged(["W"] * 25 + ["N2"] * 5, ["R"] * 25 + ["N2"] * 5)
    diag = _diag()
    diag["auto_params"]["canal_emg"] = "Chin1-Chin2"
    notes = " ".join(interpretation_notes(merged, diag))
    assert "EMG de membre" not in notes
    assert "classees eveil" not in notes


def test_params_rows_uses_readable_labels_and_marks_missing():
    rows = dict(params_rows(_diag()))
    assert rows["Backend"] == "yasa"
    assert rows["Canal EEG utilise"] == "C4-A1"
    assert rows["Canal EMG utilise"] == "non renseigne"
    assert "N2 6 (60.0%)" in rows["Repartition predite"]


def test_params_rows_empty_without_recorded_params():
    assert params_rows({"auto_method": "yasa"}) == []


def test_alignment_rows_reports_identical_hypnogram_alert():
    rows = dict(alignment_rows(_diag(identical=True)))
    assert "non valide" in rows["Alerte"]


def test_write_report_csv_contains_every_section(tmp_path):
    merged = _merged(["N2"] * 8 + ["W", "W"], ["N3"] * 8 + ["W", "W"])
    path = tmp_path / "rapport.csv"
    write_report_csv(str(path), merged, _diag(never_predicted=["N3"]))
    text = path.read_text(encoding="utf-8-sig")
    for section in (
        "# Concordance globale",
        "# Parametres du scoring automatique",
        "# Alignement des deux scorings",
        "# Metriques par stade",
        "# Matrice de confusion",
        "# Principales confusions",
        "# Interpretation",
        "# Detail par epoque",
    ):
        assert section in text
    assert "Kappa de Cohen" in text


def test_write_report_csv_detail_is_parsable_and_complete(tmp_path):
    merged = _merged(["W", "N2", "N2"], ["W", "N2", "N3"])
    path = tmp_path / "rapport.csv"
    write_report_csv(str(path), merged, _diag())
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    start = lines.index("# Detail par epoque")
    assert lines[start + 1] == "epoch,auto,manual,match"
    assert len(lines) - (start + 2) == len(merged)


def test_write_report_csv_quotes_commas_in_values(tmp_path):
    diag = _diag()
    diag["auto_params"]["candidats_eeg"] = ["C4-A1", "C3-A2"]
    path = tmp_path / "rapport.csv"
    write_report_csv(str(path), _merged(["W"], ["W"]), diag)
    text = path.read_text(encoding="utf-8-sig")
    assert '"C4-A1, C3-A2"' in text
    # A spreadsheet must still see exactly two columns on that row.
    detail = pd.read_csv(path, skiprows=text.splitlines().index("# Detail par epoque") + 1)
    assert list(detail.columns) == ["epoch", "auto", "manual", "match"]
