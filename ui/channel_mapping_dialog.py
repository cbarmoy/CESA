"""Dialog for mandatory manual channel mapping to profile sections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import tkinter as tk
from tkinter import ttk


@dataclass
class MappingResult:
    accepted: bool
    channel_mapping: Dict[str, str]  # channel -> section key, "__ignore__" for ignored


class ChannelMappingDialog:
    """Blocking dialog that asks user to map channels to sections."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        channels: Iterable[str],
        section_labels: Dict[str, str],
        prefill: Optional[Dict[str, str]] = None,
        on_configure_sections: Optional[Callable[[Dict[str, str]], Optional[Dict[str, str]]]] = None,
    ) -> None:
        self.parent = parent
        self.channels = list(channels)
        self.section_labels = dict(section_labels)
        self.prefill = dict(prefill or {})
        self.on_configure_sections = on_configure_sections
        self.result = MappingResult(accepted=False, channel_mapping={})

        self.window = tk.Toplevel(parent)
        self.window.title("Assignation manuelle des canaux")
        self.window.transient(parent)
        self.window.grab_set()
        self.window.geometry("760x520")

        self._vars: Dict[str, tk.StringVar] = {}
        self._comboboxes: Dict[str, ttk.Combobox] = {}
        self._status_vars: Dict[str, tk.StringVar] = {}
        self._choices: list[str] = []
        self._labels: Dict[str, str] = {}
        self._build()
        self.window.protocol("WM_DELETE_WINDOW", self._cancel)

    def show(self) -> MappingResult:
        self.parent.wait_window(self.window)
        return self.result

    def _build(self) -> None:
        root = ttk.Frame(self.window, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            root,
            text="Assignez chaque canal a une section d'affichage (ou Ignorer).",
        ).pack(anchor="w", pady=(0, 8))

        quick_actions = ttk.Frame(root)
        quick_actions.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(
            quick_actions,
            text="Preset: derivations uniquement",
            command=self._apply_derivations_preset,
        ).pack(side=tk.LEFT)
        if self.on_configure_sections is not None:
            ttk.Button(
                quick_actions,
                text="Configurer sections...",
                command=self._open_sections_editor,
            ).pack(side=tk.LEFT, padx=(8, 0))

        canvas = tk.Canvas(root)
        vscroll = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
        form = ttk.Frame(canvas)
        form.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=form, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        choices = ["__ignore__"] + list(self.section_labels.keys())
        labels = {"__ignore__": "Ignorer"}
        labels.update(self.section_labels)
        self._choices = choices
        self._labels = labels

        for idx, ch in enumerate(self.channels):
            ttk.Label(form, text=ch).grid(row=idx, column=0, sticky="w", padx=(0, 12), pady=2)
            initial = self.prefill.get(ch, "__ignore__")
            var = tk.StringVar(value=initial if initial in choices else "__ignore__")
            cb = ttk.Combobox(
                form,
                textvariable=var,
                values=[labels[k] for k in choices],
                state="readonly",
                width=28,
            )
            cb.grid(row=idx, column=1, sticky="w", pady=2)
            status_var = tk.StringVar()
            ttk.Label(form, textvariable=status_var, width=22).grid(row=idx, column=2, sticky="w", padx=(10, 0), pady=2)

            def _sync(_event, variable=var, widget=cb) -> None:
                # Build reverse map dynamically so section edits are reflected immediately.
                reverse_live = {self._labels[k]: k for k in self._choices}
                variable.set(reverse_live.get(widget.get(), "__ignore__"))
                self._update_row_status(ch)

            cb.bind("<<ComboboxSelected>>", _sync)
            cb.set(labels.get(var.get(), "Ignorer"))
            self._vars[ch] = var
            self._comboboxes[ch] = cb
            self._status_vars[ch] = status_var
            self._update_row_status(ch)

        actions = ttk.Frame(root)
        actions.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(actions, text="Annuler", command=self._cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(actions, text="Valider", command=self._accept).pack(side=tk.RIGHT)

    def _accept(self) -> None:
        out = {ch: var.get() for ch, var in self._vars.items()}
        self.result = MappingResult(accepted=True, channel_mapping=out)
        self.window.destroy()

    def _cancel(self) -> None:
        self.result = MappingResult(accepted=False, channel_mapping={})
        self.window.destroy()

    def _resolve_section_key(self, family: str) -> str:
        fam = family.strip().lower()
        if not fam:
            return "__ignore__"
        if fam in self._choices:
            return fam
        for key, label in self.section_labels.items():
            txt = f"{key} {label}".lower()
            if fam in txt:
                return key
        return "__ignore__"

    def _suggest_family_for_channel(self, channel_name: str) -> str:
        s = (channel_name or "").upper()
        compact = "".join(ch for ch in s if ch not in " -_/:.")

        # Ignore impedance and quality channels by default.
        if "IMP" in s or "IMPED" in s:
            return "__ignore__"
        if "ACTIVITY" in s or "AXIS" in s or "GYRO" in s or "MAGN" in s:
            return "__ignore__"

        # EOG first for E1/E2 derivations.
        if any(tag in compact for tag in ("E1M2", "E2M1", "LOCM2", "ROCM1")) or "EOG" in s:
            return "eog"

        # ECG channels.
        if any(tag in compact for tag in ("ECG", "EKG", "HEARTRATE", "FREQUENCECARDI", "FRÉQUENCECARDI")):
            return "ecg"

        # EMG/legs channels.
        if "EMG" in s or "LEFT LEG" in s or "RIGHT LEG" in s or "JAMBE" in s:
            return "emg"

        # EEG: keep only derivation-like channels for this preset.
        # This intentionally ignores mono channels like C3/O1/F3.
        eeg_deriv_tokens = ("F3M2", "F4M1", "C3M2", "C4M1", "O1M2", "O2M1")
        if any(tok in compact for tok in eeg_deriv_tokens):
            return "eeg"

        # Generic bipolar/mastoid referenced derivations (e.g. Fp1-M2, Cz-M1).
        has_mastoid_ref = ("M1" in compact or "M2" in compact)
        has_eeg_anchor = any(
            tok in compact
            for tok in ("FP", "F", "C", "P", "O", "T", "CZ", "PZ", "FZ")
        )
        if has_mastoid_ref and has_eeg_anchor:
            return "eeg"

        return "__ignore__"

    def _apply_derivations_preset(self) -> None:
        for ch, var in self._vars.items():
            family = self._suggest_family_for_channel(ch)
            section_key = self._resolve_section_key(family)
            if section_key not in self._choices:
                section_key = "__ignore__"
            var.set(section_key)
            cb = self._comboboxes.get(ch)
            if cb is not None:
                cb.set(self._labels.get(section_key, "Ignorer"))
            self._update_row_status(ch)

    def _refresh_section_choices(self) -> None:
        choices = ["__ignore__"] + list(self.section_labels.keys())
        labels = {"__ignore__": "Ignorer"}
        labels.update(self.section_labels)
        self._choices = choices
        self._labels = labels

        for ch, var in self._vars.items():
            current = str(var.get() or "").strip()
            if current not in choices:
                current = "__ignore__"
                var.set(current)
            cb = self._comboboxes.get(ch)
            if cb is None:
                continue
            cb.configure(values=[labels[k] for k in choices])
            cb.set(labels.get(current, "Ignorer"))
            self._update_row_status(ch)

    def _open_sections_editor(self) -> None:
        if self.on_configure_sections is None:
            return
        try:
            updated = self.on_configure_sections(dict(self.section_labels))
        except Exception:
            updated = None
        if not updated:
            return
        self.section_labels = dict(updated)
        self._refresh_section_choices()

    def _update_row_status(self, channel: str) -> None:
        var = self._vars.get(channel)
        status_var = self._status_vars.get(channel)
        if var is None or status_var is None:
            return
        key = str(var.get() or "").strip()
        label = self._labels.get(key, "Ignorer")
        if key == "__ignore__":
            status_var.set("Affectation: Ignorer")
        else:
            status_var.set(f"Affectation: {label}")

