"""Unified dialog to select a recording file and choose the loading mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from core.raw_loader import recording_filetypes_for_dialog


@dataclass
class OpenDatasetSelection:
    """User choices captured by the dialog."""

    edf_path: str
    mode: str  # "raw" or "precomputed"
    ms_path: Optional[str]
    precompute_action: str  # "build" or "existing"


class OpenDatasetDialog:
    """Gather recording path, navigation mode, and optional multiscale settings."""

    def __init__(self, parent: tk.Tk | tk.Toplevel) -> None:
        self.parent = parent
        self.result: Optional[OpenDatasetSelection] = None

        self.window = tk.Toplevel(parent)
        self.window.title("Ouvrir un enregistrement")
        self.window.transient(parent)
        self.window.grab_set()
        self.window.resizable(False, False)

        self.edf_path_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="raw")
        self.ms_path_var = tk.StringVar()
        self.precompute_action_var = tk.StringVar(value="build")

        self._ms_path_user_override = False

        self._build_ui()

        self.window.protocol("WM_DELETE_WINDOW", self._cancel)
        # Focus the dialog after layout
        self.window.after(50, self.window.focus_force)

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        container = ttk.Frame(self.window, padding=15)
        container.pack(fill=tk.BOTH, expand=True)

        # Recording selection -------------------------------------------
        edf_group = ttk.LabelFrame(container, text="Fichier d'enregistrement")
        edf_group.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        edf_entry = ttk.Entry(edf_group, textvariable=self.edf_path_var, width=50)
        edf_entry.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="ew")
        ttk.Button(
            edf_group,
            text="Parcourir...",
            command=self._browse_edf,
        ).grid(row=0, column=1, padx=(0, 10), pady=10)

        info_label = ttk.Label(
            edf_group,
            text="Choisissez le fichier de signal a analyser (EDF/BDF/FIF...).",
            foreground="#555555",
        )
        info_label.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="w")

        edf_group.columnconfigure(0, weight=1)

        # Mode selection -------------------------------------------------
        mode_group = ttk.LabelFrame(container, text="Mode de lecture")
        mode_group.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        ttk.Radiobutton(
            mode_group,
            text="Mode standard (lecture directe depuis l'EDF)",
            value="raw",
            variable=self.mode_var,
            command=self._toggle_mode,
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")

        ttk.Radiobutton(
            mode_group,
            text="Navigation rapide (fichier pré-calculé)",
            value="precomputed",
            variable=self.mode_var,
            command=self._toggle_mode,
        ).grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # Precomputed options -------------------------------------------
        self.precompute_frame = ttk.Frame(mode_group)
        self.precompute_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=(5, 10), sticky="ew")

        ttk.Label(self.precompute_frame, text="Dossier de navigation rapide (Zarr)").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )

        self.ms_entry = ttk.Entry(self.precompute_frame, textvariable=self.ms_path_var, width=50)
        self.ms_entry.grid(row=1, column=0, padx=(0, 5), pady=5, sticky="ew")

        ttk.Button(
            self.precompute_frame,
            text="Sélectionner...",
            command=self._browse_ms_path,
        ).grid(row=1, column=1, padx=(0, 0), pady=5)

        action_frame = ttk.Frame(self.precompute_frame)
        action_frame.grid(row=2, column=0, columnspan=2, pady=(5, 0), sticky="w")

        ttk.Radiobutton(
            action_frame,
            text="Créer / mettre à jour automatiquement",
            value="build",
            variable=self.precompute_action_var,
        ).pack(anchor="w")

        ttk.Radiobutton(
            action_frame,
            text="Utiliser un fichier existant",
            value="existing",
            variable=self.precompute_action_var,
        ).pack(anchor="w")

        helper = ttk.Label(
            self.precompute_frame,
            text="Le fichier sera généré si besoin (quelques minutes selon la taille).",
            foreground="#555555",
        )
        helper.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.precompute_frame.columnconfigure(0, weight=1)

        # Action buttons -------------------------------------------------
        buttons = ttk.Frame(container)
        buttons.pack(fill=tk.X, pady=(0, 0))

        ttk.Button(buttons, text="Annuler", command=self._cancel).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(buttons, text="Ouvrir", command=self._confirm).pack(side=tk.RIGHT)

        # Bindings -------------------------------------------------------
        self.edf_path_var.trace_add("write", self._on_edf_change)
        self.ms_path_var.trace_add("write", self._on_ms_change)

        self._toggle_mode()  # initialise state

    # ------------------------------------------------------------------
    def show(self) -> Optional[OpenDatasetSelection]:
        self.parent.wait_window(self.window)
        return self.result

    # ------------------------------------------------------------------
    def _browse_edf(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.window,
            title="Selectionner un enregistrement",
            filetypes=recording_filetypes_for_dialog(),
        )
        if path:
            self.edf_path_var.set(path)

    def _browse_ms_path(self) -> None:
        path = filedialog.askdirectory(
            parent=self.window,
            title="Sélectionner un dossier de navigation rapide",
        )
        if path:
            self._ms_path_user_override = True
            self.ms_path_var.set(path)

    def _on_edf_change(self, *_args) -> None:
        if self.mode_var.get() != "precomputed":
            return
        if self._ms_path_user_override:
            return
        edf_path = Path(self.edf_path_var.get().strip())
        if not edf_path.name:
            return
        default_ms = edf_path.with_suffix("") / "_ms"
        self.ms_path_var.set(str(default_ms))

    def _on_ms_change(self, *_args) -> None:
        # If the change originates from user typing, mark override
        if self.window.focus_get() is self.ms_entry:
            self._ms_path_user_override = True

    def _toggle_mode(self) -> None:
        is_precomputed = self.mode_var.get() == "precomputed"
        state = "normal" if is_precomputed else "disabled"
        for child in self.precompute_frame.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass
        if is_precomputed:
            self._on_edf_change()

    def _confirm(self) -> None:
        edf_path = self.edf_path_var.get().strip()
        if not edf_path:
            messagebox.showwarning(
                "Fichier manquant",
                "Veuillez selectionner un fichier d'enregistrement.",
                parent=self.window,
            )
            return
        if not Path(edf_path).exists():
            messagebox.showwarning(
                "Fichier introuvable",
                "Le fichier selectionne n'existe pas.",
                parent=self.window,
            )
            return

        mode = self.mode_var.get()
        ms_path: Optional[str] = None
        precompute_action = "existing"

        if mode == "precomputed":
            ms_candidate = self.ms_path_var.get().strip()
            if not ms_candidate:
                messagebox.showwarning(
                    "Chemin requis",
                    "Indiquez un dossier pour le fichier de navigation rapide.",
                    parent=self.window,
                )
                return
            ms_path = ms_candidate
            precompute_action = self.precompute_action_var.get()

        self.result = OpenDatasetSelection(
            edf_path=edf_path,
            mode=mode,
            ms_path=ms_path,
            precompute_action=precompute_action,
        )
        self.window.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.window.destroy()


