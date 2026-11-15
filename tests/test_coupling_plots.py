"""Tests for the circular coupling figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from CESA import coupling_plots as cp


def cohort() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(2)
    for i in range(8):
        subj = f"S{i:03d}"
        for group, mean in (("AVANT", -0.3), ("APRES", 0.2)):
            rows.append({
                "subject": f"{subj}_{'Av' if group == 'AVANT' else 'Ap'}",
                "group": group,
                "channel": "C3-M2",
                "preferred_phase_rad": float(mean + rng.normal(0, 0.15)),
                "preferred_phase_deg": 0.0,
                "resultant_length": 0.6,
                "n_coupled": 40,
            })
    table = pd.DataFrame(rows)
    table["preferred_phase_deg"] = np.degrees(table["preferred_phase_rad"])
    return table


def test_participant_key_strips_condition():
    assert cp.participant_key("S030_Ap") == "S030"
    assert cp.participant_key("S030_Av") == "S030"


def test_phase_rose_draws_arrow_above_threshold():
    rng = np.random.default_rng(0)
    phases = rng.vonmises(0.4, 3.0, 40)
    fig = cp.plot_phase_rose(phases, channel="Cz")
    assert fig.axes
    assert fig.axes[0].name == "polar"


def test_phase_rose_empty_raises():
    with pytest.raises(ValueError):
        cp.plot_phase_rose([])


def test_group_and_shift_figures(tmp_path):
    table = cohort()
    fig = cp.plot_group_phases(table)
    assert len(fig.axes) == 1
    fig2 = cp.plot_phase_shift(table)
    assert len(fig2.axes) == 1
    deltas = cp.paired_phase_shifts(table)
    assert deltas.size == 8


def test_save_coupling_figures_writes_both(tmp_path):
    written = cp.save_coupling_figures(cohort(), tmp_path)
    names = {p.split("\\")[-1].split("/")[-1] for p in written}
    assert "coupling_phases.png" in names
    assert "coupling_decalage.png" in names


def test_save_skips_shift_without_both_groups(tmp_path):
    table = cohort()
    written = cp.save_coupling_figures(table[table["group"] == "AVANT"], tmp_path)
    names = {p.split("\\")[-1].split("/")[-1] for p in written}
    assert "coupling_phases.png" in names
    assert "coupling_decalage.png" not in names
