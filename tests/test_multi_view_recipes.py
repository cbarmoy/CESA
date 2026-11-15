"""Tests for multi-view filter recipes (duplicate-channel scoring)."""

from __future__ import annotations

from CESA.qt_viewer.multi_view_recipes import (
    get_recipe,
    get_recipes,
    make_lane_label,
    recipes_for_channel_type,
)


def test_builtin_recipes_non_empty():
    recipes = get_recipes()
    assert len(recipes) >= 5
    ids = {r.recipe_id for r in recipes}
    assert "n2_scoring" in ids
    assert "so_spindle" in ids
    assert "eog_rem" in ids


def test_n2_recipe_has_three_distinct_lanes():
    r = get_recipe("n2_scoring")
    assert r is not None
    assert len(r.lanes) == 3
    names = [ln.short_name for ln in r.lanes]
    assert "Standard" in names
    assert "Spindles" in names
    assert "K-complexe" in names
    colors = [ln.color for ln in r.lanes]
    assert len(set(colors)) == 3
    assert sum(1 for ln in r.lanes if ln.is_primary) == 1


def test_recipes_filtered_by_channel_type():
    eeg = recipes_for_channel_type("eeg")
    eog = recipes_for_channel_type("eog")
    assert all("eeg" in r.channel_types or "generic" in r.channel_types for r in eeg)
    assert any(r.recipe_id == "eog_rem" for r in eog)
    assert all(r.recipe_id != "eog_rem" for r in eeg)


def test_make_lane_label():
    assert make_lane_label("C3-M2", "Spindles") == "C3-M2 · Spindles"
