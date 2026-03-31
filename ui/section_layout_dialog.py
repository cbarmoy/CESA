"""Dialog to edit profile signal sections (layout + colors)."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List

import tkinter as tk
from tkinter import colorchooser, messagebox, ttk

from CESA.profile_schema import SignalSection


HEX_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{6})$")


@dataclass
class SectionLayoutResult:
    accepted: bool
    sections: List[SignalSection]


class SectionLayoutDialog:
    """Blocking dialog for configuring section names, sizes and colors."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        sections: List[SignalSection],
    ) -> None:
        self.parent = parent
        self._rows: List[Dict[str, object]] = []
        self._counter = 1
        self.result = SectionLayoutResult(accepted=False, sections=[])

        self.window = tk.Toplevel(parent)
        self.window.title("Configurer sections du profil")
        self.window.transient(parent)
        self.window.grab_set()
        self.window.geometry("980x560")

        self._build()
        for section in sections:
            self._add_row(section)
        if not sections:
            self._add_row(
                SignalSection(
                    key="eeg",
                    label="EEG",
                    ratio=5.0,
                    signal_type="eeg",
                    enabled=True,
                    color_palette=["#3B82F6", "#22C55E"],
                )
            )
        self.window.protocol("WM_DELETE_WINDOW", self._cancel)

    def show(self) -> SectionLayoutResult:
        self.parent.wait_window(self.window)
        return self.result

    def _build(self) -> None:
        root = ttk.Frame(self.window, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            root,
            text="Définissez le nombre de sections, leur nom, leur taille (ratio) et leur palette de couleurs.",
        ).pack(anchor="w", pady=(0, 8))

        header = ttk.Frame(root)
        header.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(header, text="Activé", width=8).grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Nom section", width=22).grid(row=0, column=1, sticky="w")
        ttk.Label(header, text="Type", width=14).grid(row=0, column=2, sticky="w")
        ttk.Label(header, text="Taille box (ratio)", width=16).grid(row=0, column=3, sticky="w")
        ttk.Label(header, text="Palette (#RRGGBB, ...)", width=42).grid(row=0, column=4, sticky="w")
        ttk.Label(header, text="", width=10).grid(row=0, column=5, sticky="w")

        canvas = tk.Canvas(root)
        vscroll = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
        self.form = ttk.Frame(canvas)
        self.form.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.form, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        actions = ttk.Frame(root)
        actions.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(actions, text="+ Ajouter section", command=self._add_empty_row).pack(side=tk.LEFT)
        ttk.Button(actions, text="Annuler", command=self._cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(actions, text="Valider", command=self._accept).pack(side=tk.RIGHT)

    def _next_key(self) -> str:
        key = f"section_{self._counter}"
        self._counter += 1
        return key

    def _normalize_palette(self, text: str) -> List[str]:
        out: List[str] = []
        for token in str(text or "").split(","):
            color = token.strip()
            if not color:
                continue
            if not color.startswith("#"):
                color = f"#{color}"
            if HEX_COLOR_RE.match(color):
                out.append(color.upper())
        return out

    def _append_color(self, palette_var: tk.StringVar) -> None:
        picked = colorchooser.askcolor(parent=self.window)[1]
        if not picked:
            return
        current = self._normalize_palette(palette_var.get())
        current.append(str(picked).upper())
        palette_var.set(", ".join(current))

    def _add_empty_row(self) -> None:
        self._add_row(
            SignalSection(
                key=self._next_key(),
                label=f"Section {len(self._rows) + 1}",
                ratio=1.0,
                signal_type="eeg",
                enabled=True,
                color_palette=["#3B82F6", "#22C55E"],
            )
        )

    def _remove_row(self, row: Dict[str, object]) -> None:
        frame = row.get("frame")
        if frame is not None:
            try:
                frame.destroy()
            except Exception:
                pass
        self._rows = [r for r in self._rows if r is not row]
        self._regrid_rows()

    def _regrid_rows(self) -> None:
        for idx, row in enumerate(self._rows):
            frame = row.get("frame")
            if frame is not None:
                frame.grid(row=idx, column=0, sticky="ew", pady=2)

    def _add_row(self, section: SignalSection) -> None:
        frame = ttk.Frame(self.form)
        row_idx = len(self._rows)
        frame.grid(row=row_idx, column=0, sticky="ew", pady=2)

        key = str(section.key or "").strip() or self._next_key()
        label_var = tk.StringVar(value=str(section.label or key))
        type_var = tk.StringVar(value=str(section.signal_type or "eeg").lower())
        ratio_var = tk.StringVar(value=f"{float(section.ratio):.2f}")
        palette_var = tk.StringVar(value=", ".join(section.color_palette or []))
        enabled_var = tk.BooleanVar(value=bool(section.enabled))

        ttk.Checkbutton(frame, variable=enabled_var).grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(frame, textvariable=label_var, width=24).grid(row=0, column=1, sticky="w", padx=(0, 8))
        ttk.Combobox(
            frame,
            textvariable=type_var,
            values=["eeg", "eog", "emg", "ecg", "custom"],
            state="readonly",
            width=12,
        ).grid(row=0, column=2, sticky="w", padx=(0, 8))
        ttk.Entry(frame, textvariable=ratio_var, width=14).grid(row=0, column=3, sticky="w", padx=(0, 8))
        ttk.Entry(frame, textvariable=palette_var, width=44).grid(row=0, column=4, sticky="w", padx=(0, 4))
        ttk.Button(frame, text="Couleur+", command=lambda v=palette_var: self._append_color(v)).grid(
            row=0, column=5, sticky="w", padx=(0, 4)
        )
        ttk.Button(frame, text="Supprimer", command=lambda r=None: None).grid(row=0, column=6, sticky="w")
        remove_btn = frame.grid_slaves(row=0, column=6)[0]
        remove_btn.configure(command=lambda row_ref=None: None)

        row = {
            "frame": frame,
            "key": key,
            "label_var": label_var,
            "type_var": type_var,
            "ratio_var": ratio_var,
            "palette_var": palette_var,
            "enabled_var": enabled_var,
        }
        remove_btn.configure(command=lambda row_ref=row: self._remove_row(row_ref))
        self._rows.append(row)

    def _accept(self) -> None:
        if not self._rows:
            messagebox.showwarning("Sections", "Ajoutez au moins une section.", parent=self.window)
            return
        out: List[SignalSection] = []
        used_keys: set[str] = set()
        for idx, row in enumerate(self._rows):
            key = str(row["key"]).strip() or f"section_{idx + 1}"
            if key in used_keys:
                key = f"{key}_{idx + 1}"
            used_keys.add(key)
            label = str(row["label_var"].get()).strip() or key
            stype = str(row["type_var"].get()).strip().lower() or "eeg"
            try:
                ratio = float(str(row["ratio_var"].get()).strip())
            except Exception:
                ratio = 1.0
            ratio = max(0.2, ratio)
            palette = self._normalize_palette(row["palette_var"].get())
            out.append(
                SignalSection(
                    key=key,
                    label=label,
                    ratio=ratio,
                    signal_type=stype,
                    enabled=bool(row["enabled_var"].get()),
                    color_palette=palette,
                )
            )
        if not any(bool(s.enabled) for s in out):
            messagebox.showwarning("Sections", "Activez au moins une section.", parent=self.window)
            return
        self.result = SectionLayoutResult(accepted=True, sections=out)
        self.window.destroy()

    def _cancel(self) -> None:
        self.result = SectionLayoutResult(accepted=False, sections=[])
        self.window.destroy()

