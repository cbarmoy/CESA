"""Unit tests for the SO-spindle coupling figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from CESA import coupling_plots as cp


@pytest.fixture(autouse=True)
def _agg_backend():
    import matplotlib

    matplotlib.use("Agg", force=False)


def cohort(n_subjects: int = 8, *, hide: int = 0) -> pd.DataFrame:
    """Batch-shaped table: one row per recording x channel, two conditions."""
    rng = np.random.default_rng(4)
    rows = []
    for i in range(n_subjects):
        subject = f"S{30 + i:03d}"
        for condition, suffix, centre in (
            ("AVANT", "Av", -0.5), ("APRES", "Ap", -0.1),
        ):
            for channel in ("C3-M2", "F3-M2"):
                phase = float(rng.vonmises(centre, 4.0))
                rows.append({
                    "subject": f"{subject}_{suffix}",
                    "file": f"{subject}_{suffix}.edf",
                    "group": condition,
                    "channel": channel,
                    "n_coupled": 40,
                    "preferred_phase_rad": phase,
                    "preferred_phase_deg": np.degrees(phase),
                    "resultant_length": 0.45,
                    "rayleigh_p": 0.01,
                })
    table = pd.DataFrame(rows)
    for index in range(hide):
        table.loc[index, ["preferred_phase_rad", "resultant_length"]] = np.nan
    return table


class TestParticipantKey:
    def test_strips_the_condition_token(self):
        assert cp.participant_key("S030_Ap") == "S030"
        assert cp.participant_key("S030_AV") == "S030"
        assert cp.participant_key("S030_AP_nuit") == "S030_nuit"

    def test_keeps_a_plain_identifier(self):
        assert cp.participant_key("sujet12") == "sujet12"
        assert cp.participant_key("S030") == "S030"


class TestPhaseRose:
    def test_draws_the_mean_vector_above_the_threshold(self):
        rng = np.random.default_rng(1)
        phases = rng.vonmises(0.2, 3.0, 40)
        fig = cp.plot_phase_rose(phases, channel="F3-M2")
        ax = fig.axes[0]
        assert ax.name == "polar"
        assert ax.patches  # histogram bars
        texts = " ".join(t.get_text() for t in fig.findobj(match=_is_text))
        assert "n = 40" in texts
        assert "phase préférée" in texts

    def test_hides_the_mean_vector_below_the_threshold(self):
        fig = cp.plot_phase_rose([0.1, 0.2, -0.1], channel="F3-M2")
        texts = " ".join(t.get_text() for t in fig.findobj(match=_is_text))
        assert "non publiée" in texts
        assert "phase préférée =" not in texts

    def test_uses_the_provided_stats(self):
        stats = {
            "n": 30, "preferred_phase_rad": 0.5, "preferred_phase_deg": 28.6,
            "resultant_length": 0.7, "rayleigh_p": 1e-4,
        }
        fig = cp.plot_phase_rose(np.zeros(30), channel="C3-M2", stats=stats)
        texts = " ".join(t.get_text() for t in fig.findobj(match=_is_text))
        assert "28.6°" in texts

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="aucune phase"):
            cp.plot_phase_rose([])

    def test_non_finite_values_are_dropped(self):
        fig = cp.plot_phase_rose([0.1, np.nan, 0.2, np.inf] * 10)
        texts = " ".join(t.get_text() for t in fig.findobj(match=_is_text))
        assert "n = 20" in texts

    def test_dark_palette_changes_the_background(self):
        light = cp.plot_phase_rose(np.zeros(20))
        dark = cp.plot_phase_rose(np.zeros(20), dark=True)
        assert light.get_facecolor() != dark.get_facecolor()


class TestGroupPhases:
    def test_one_polar_axes_per_channel(self):
        fig = cp.plot_group_phases(cohort())
        assert len(fig.axes) == 2
        assert all(ax.name == "polar" for ax in fig.axes)

    def test_reports_the_rows_without_a_published_phase(self):
        fig = cp.plot_group_phases(cohort(hide=2))
        texts = " ".join(t.get_text() for t in fig.findobj(match=_is_text))
        assert "sans phase publiée" in texts

    def test_small_group_does_not_get_a_mean_vector(self):
        fig = cp.plot_group_phases(cohort(n_subjects=2))
        texts = " ".join(t.get_text() for t in fig.findobj(match=_is_text))
        assert "non tracée" in texts

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="preferred_phase_rad"):
            cp.plot_group_phases(pd.DataFrame({"channel": ["C3"]}))


class TestPairedShift:
    def test_pairs_the_two_nights_of_each_subject(self):
        table = cohort(n_subjects=6)
        shifts = cp.paired_phase_shifts(table[table["channel"] == "C3-M2"])
        assert shifts.size == 6
        assert np.all(np.abs(shifts) <= np.pi)

    def test_withheld_phases_are_dropped_from_the_pairing(self):
        table = cohort(n_subjects=6)
        c3 = table[table["channel"] == "C3-M2"].copy()
        c3.loc[c3.index[0], "preferred_phase_rad"] = np.nan
        assert cp.paired_phase_shifts(c3).size == 5

    def test_single_condition_yields_no_pair(self):
        table = cohort(n_subjects=6)
        only_before = table[table["group"] == "AVANT"]
        assert cp.paired_phase_shifts(only_before).size == 0

    def test_shift_figure_reports_the_rayleigh_test(self):
        fig = cp.plot_phase_shift(cohort())
        texts = " ".join(t.get_text() for t in fig.findobj(match=_is_text))
        assert "décalage moyen" in texts
        assert "Rayleigh p" in texts

    def test_shift_figure_without_pairs_says_so(self):
        table = cohort()
        table = table[table["group"] == "AVANT"]
        fig = cp.plot_phase_shift(table)
        texts = " ".join(t.get_text() for t in fig.findobj(match=_is_text))
        assert "aucune paire" in texts


class TestSaveCouplingFigures:
    def test_writes_both_figures(self, tmp_path):
        written = cp.save_coupling_figures(cohort(), tmp_path)
        assert [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in written] == [
            "coupling_phases.png", "coupling_decalage.png",
        ]
        for path in written:
            assert (tmp_path / path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]).stat().st_size > 0

    def test_single_condition_skips_the_shift_figure(self, tmp_path):
        table = cohort()
        written = cp.save_coupling_figures(table[table["group"] == "AVANT"], tmp_path)
        assert len(written) == 1
        assert written[0].endswith("coupling_phases.png")

    def test_reads_a_csv_path(self, tmp_path):
        csv_path = tmp_path / "batch.csv"
        cohort().to_csv(csv_path, index=False)
        written = cp.save_coupling_figures(str(csv_path), tmp_path, prefix="essai")
        assert any("essai_phases.png" in p for p in written)

    def test_unusable_table_writes_nothing(self, tmp_path):
        empty = pd.DataFrame({"channel": [], "subject": [], "group": []})
        assert cp.save_coupling_figures(empty, tmp_path) == []


def _is_text(artist) -> bool:
    from matplotlib.text import Text

    return isinstance(artist, Text)
