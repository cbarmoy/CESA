"""Startup mode selection dialog for PSG viewers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from core.providers import PrecomputedProvider, SignalProvider


@dataclass(frozen=True)
class ModeSelection:
    mode: str
    provider: Optional[SignalProvider]
    source_path: Optional[Path]


class ModeSelector:
    """Minimal Tk dialog to choose between precomputed and lazy data modes."""

    def __init__(
        self,
        parent: tk.Tk,
        *,
        default_ms_path: str | Path | None = None,
        lazy_available: bool = False,
    ) -> None:
        self._parent = parent
        self._default_ms_path = Path(default_ms_path) if default_ms_path else None
        self._lazy_available = bool(lazy_available)

    def choose(self) -> ModeSelection:
        if self._default_ms_path is not None:
            provider = PrecomputedProvider(self._default_ms_path)
            return ModeSelection(mode="precomputed", provider=provider, source_path=self._default_ms_path)

        mode, path = self._show_dialog()
        if mode == "precomputed" and path is not None:
            provider = PrecomputedProvider(path)
            return ModeSelection(mode="precomputed", provider=provider, source_path=path)
        return ModeSelection(mode=mode, provider=None, source_path=path)
    
    def get_selection(self) -> tuple[Optional[str], Optional[Path]]:
        """
        Lightweight version for UI integration: returns (mode, ms_path).
        Returns (None, None) if user cancels.
        """
        try:
            mode, path = self._show_dialog()
            return (mode, path)
        except RuntimeError:
            # User cancelled
            return (None, None)

    # ------------------------------------------------------------------
    def _show_dialog(self) -> tuple[str, Optional[Path]]:
        dialog = tk.Toplevel(self._parent)
        dialog.title("Sélection du mode de données")
        dialog.transient(self._parent)
        dialog.grab_set()
        dialog.resizable(False, False)

        container = ttk.Frame(dialog, padding=16)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)

        title = ttk.Label(container, text="Mode de chargement des données", font=("Segoe UI", 14, "bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        subtitle = ttk.Label(
            container, 
            text="Choisissez comment vous souhaitez charger votre fichier EEG :",
            font=("Segoe UI", 9)
        )
        subtitle.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 16))

        mode_var = tk.StringVar(value="precomputed")

        # Cadre pour Navigation Rapide
        fast_frame = ttk.LabelFrame(container, text="⚡ Navigation Rapide (Recommandé)", padding=10)
        fast_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        
        precomputed_radio = ttk.Radiobutton(
            fast_frame,
            text="Activer la navigation instantanée",
            value="precomputed",
            variable=mode_var,
        )
        precomputed_radio.grid(row=0, column=0, columnspan=3, sticky="w")
        
        fast_desc = ttk.Label(
            fast_frame,
            text="• Déplacement et zoom ultra-rapides\n• Idéal pour les longues sessions d'analyse\n• Nécessite une préparation initiale (quelques minutes)",
            foreground="#059669",
            font=("Segoe UI", 8)
        )
        fast_desc.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # Cadre pour Mode Standard
        standard_frame = ttk.LabelFrame(container, text="📂 Modes dynamiques", padding=10)
        standard_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 16))

        options: list[tuple[str, str, str]] = [
            ("Chargement classique", "raw", "• Chargement direct sans préparation\n• Adapté aux fichiers courts ou analyse ponctuelle"),
        ]
        if self._lazy_available:
            options.insert(0, (
                "Calcul à la volée (Lazy)",
                "lazy",
                "• Min/Max calculés à la demande\n• Idéal pour tester sans pré-calcul\n• Peut prendre quelques secondes la première fois",
            ))

        start_row = 0
        for label, value, desc in options:
            ttk.Radiobutton(standard_frame, text=label, value=value, variable=mode_var).grid(
                row=start_row, column=0, columnspan=3, sticky="w"
            )
            ttk.Label(
                standard_frame,
                text=desc,
                foreground="#6b7280" if value == "raw" else "#2563eb",
                font=("Segoe UI", 8),
                justify="left",
            ).grid(row=start_row + 1, column=0, columnspan=3, sticky="w", pady=(4, 4))
            start_row += 2

        ttk.Separator(container, orient="horizontal").grid(row=4, column=0, columnspan=3, sticky="ew", pady=12)

        # Section fichier pré-calculé
        path_label = ttk.Label(container, text="Fichier de navigation rapide existant (optionnel) :", font=("Segoe UI", 9, "bold"))
        path_label.grid(row=5, column=0, columnspan=3, sticky="w", pady=(0, 4))
        
        path_var = tk.StringVar(value="")

        path_info = ttk.Label(
            container,
            text="💡 Première utilisation ? Laissez ce champ vide !\n"
                 "CESA vous proposera de créer le fichier automatiquement.\n\n"
                 "Vous avez déjà un fichier ? Sélectionnez-le avec 'Parcourir...'",
            foreground="#059669",
            font=("Segoe UI", 8),
            justify="left"
        )
        path_info.grid(row=6, column=0, columnspan=3, sticky="w", pady=(0, 8))

        entry = ttk.Entry(container, textvariable=path_var, width=40)
        entry.grid(row=7, column=0, columnspan=2, sticky="ew", pady=4)

        def _browse() -> None:
            directory = filedialog.askdirectory(parent=dialog, title="Sélectionner le fichier de navigation rapide")
            if directory:
                path_var.set(directory)

        browse_btn = ttk.Button(container, text="Parcourir...", command=_browse)
        browse_btn.grid(row=7, column=2, sticky="ew", padx=(6, 0))

        ttk.Separator(container, orient="horizontal").grid(row=8, column=0, columnspan=3, sticky="ew", pady=12)

        error_var = tk.StringVar(value="")
        error_label = ttk.Label(container, textvariable=error_var, foreground="#dc2626")
        error_label.grid(row=9, column=0, columnspan=3, sticky="w", pady=(0, 8))

        resolved_path: Optional[Path] = None
        resolved_mode: Optional[str] = None

        def _confirm() -> None:
            nonlocal resolved_path, resolved_mode
            selected = mode_var.get()

            if selected == "raw":
                resolved_mode = "raw"
                resolved_path = None
                dialog.destroy()
                return

            if selected == "lazy":
                resolved_mode = "lazy"
                resolved_path = None
                dialog.destroy()
                return

            if selected == "precomputed":
                value = path_var.get().strip()
                if value:
                    candidate = Path(value)
                    if not candidate.exists():
                        error_var.set("Dossier introuvable. Laissez vide pour créer automatiquement.")
                        return
                    if not (candidate / ".zattrs").exists() and not (candidate / "levels").exists():
                        error_var.set("Ce dossier ne semble pas être un fichier de navigation rapide valide.\nLaissez vide pour en créer un nouveau.")
                        return
                    resolved_path = candidate
                else:
                    resolved_path = None
                resolved_mode = "precomputed"
                dialog.destroy()
                return

            resolved_mode = selected
            resolved_path = path_var.get().strip() or None
            dialog.destroy()

        buttons = ttk.Frame(container)
        buttons.grid(row=10, column=0, columnspan=3, sticky="e")
        ttk.Button(buttons, text="Continuer", command=_confirm).grid(row=0, column=0, padx=(0, 4))

        def _cancel() -> None:
            dialog.destroy()

        ttk.Button(buttons, text="Annuler", command=_cancel).grid(row=0, column=1)

        entry.focus_set()
        dialog.bind("<Return>", lambda _evt: _confirm())
        dialog.bind("<Escape>", lambda _evt: _cancel())
        
        self._parent.wait_window(dialog)

        # Check if dialog was cancelled (mode wasn't set)
        if resolved_mode is None:
            raise RuntimeError("Sélection du mode annulée par l'utilisateur")
        
        # Return (mode, path) - path can be None for raw mode
        return (resolved_mode, resolved_path)


