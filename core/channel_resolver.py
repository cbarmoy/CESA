"""Helpers to resolve preferred channels across naming variants."""

from __future__ import annotations

from typing import Iterable, Sequence


def normalize_channel_name(name: str) -> str:
    s = (name or "").upper()
    for ch in (" ", "-", "_", "/", ":", "."):
        s = s.replace(ch, "")
    return s


def resolve_aliases(available_channels: Sequence[str], aliases: Iterable[str]) -> str | None:
    if not available_channels:
        return None
    normalized_map: dict[str, str] = {}
    for ch in available_channels:
        key = normalize_channel_name(ch)
        if key and key not in normalized_map:
            normalized_map[key] = ch

    for alias in aliases:
        candidate = normalized_map.get(normalize_channel_name(alias))
        if candidate:
            return candidate
    return None


def select_preferred_channels(
    available_channels: Sequence[str],
    preferred_alias_groups: Sequence[Iterable[str]],
    *,
    max_channels: int = 8,
) -> list[str]:
    """Select channels by preferred groups, resilient to naming differences."""
    chosen: list[str] = []
    for aliases in preferred_alias_groups:
        resolved = resolve_aliases(available_channels, aliases)
        if resolved and resolved not in chosen:
            chosen.append(resolved)
        if len(chosen) >= max_channels:
            break
    return chosen[:max_channels]
