"""
CESA v0.0beta1.1 - Mathematical Formulas Display Window
===============================================

Module pour afficher les formules mathématiques utilisées dans CESA.
Développé pour l'Unité Neuropsychologie du Stress (IRBA).

Fonctionnalités:
- Affichage des formules mathématiques avec formatage professionnel
- Support des équations EEG et analyses spectrales
- Interface graphique simple et intuitive

Formules incluses:
- Analyse spectrale (FFT, PSD, Welch)
- Détection de stades de sommeil
- Analyses statistiques (cohérence, corrélation)
- Filtres Butterworth

Auteur: Côme Barmoy (Unité Neuropsychologie du Stress - IRBA)
Version: 0.0beta1.1
Date: 2025-09-26
"""

import tkinter as tk
from tkinter import ttk, messagebox
import math


def show_math_formulas_window(parent=None):
    """Affiche une fenêtre avec les formules mathématiques utilisées dans CESA."""
    try:
        window = tk.Toplevel(parent) if parent else tk.Tk()
        window.title("🧮 Formules Mathématiques - CESA v0.0beta1.1")
        window.geometry("900x700")

        # Créer un notebook pour organiser les formules par catégorie
        notebook = ttk.Notebook(window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Style pour les formules
        style = ttk.Style()
        style.configure("Formula.TLabel", font=("Cambria Math", 12))
        style.configure("Title.TLabel", font=("Arial", 14, "bold"))

        # === Onglet 1: Analyse Spectrale ===
        frame_spectral = ttk.Frame(notebook)
        notebook.add(frame_spectral, text="Analyse Spectrale")

        # Titre
        title_label = ttk.Label(frame_spectral, text="Analyse Spectrale", style="Title.TLabel")
        title_label.pack(pady=(10, 5))

        # Frame pour le contenu avec scrollbar
        content_frame = ttk.Frame(frame_spectral)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        # Canvas et scrollbar
        canvas = tk.Canvas(content_frame)
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Placement des éléments
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Fonction pour créer des labels de formules
        def create_formula_label(parent, formula, description=""):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=2)

            formula_label = ttk.Label(frame, text=formula, style="Formula.TLabel", justify=tk.CENTER)
            formula_label.pack()

            if description:
                desc_label = ttk.Label(frame, text=description, font=("Arial", 9))
                desc_label.pack()

            return frame

        # Formules d'analyse spectrale
        formulas = [
            ("FFT (Transformée de Fourier Rapide):", "X[k] = Σ_{n=0}^{N-1} x[n] ⋅ e^{-j2πkn/N}"),
            ("PSD (Densité Spectrale de Puissance):", "PSD(f) = |FFT(x)|² / N"),
            ("Méthode de Welch:", "PSD_{Welch}(f) = (1/M) Σ_{i=1}^M |FFT(x_i)|²"),
            ("Pouvoir Spectral par Bande:", "P_{bande} = Σ_{f∈bande} PSD(f) ⋅ Δf"),
            ("Fréquence Dominante:", "f_{dom} = argmax_f PSD(f)"),
            ("Centroïde Spectral:", "f_c = (Σ_f f ⋅ PSD(f)) / (Σ_f PSD(f))"),
        ]

        for title, formula in formulas:
            sub_frame = ttk.Frame(scrollable_frame)
            sub_frame.pack(fill=tk.X, pady=5)

            title_label = ttk.Label(sub_frame, text=title, font=("Arial", 10, "bold"))
            title_label.pack(anchor=tk.W)

            formula_label = ttk.Label(sub_frame, text=formula, font=("Cambria Math", 11), justify=tk.CENTER)
            formula_label.pack(pady=(0, 5))

        # === Onglet 2: Détection de Stades ===
        frame_stages = ttk.Frame(notebook)
        notebook.add(frame_stages, text="Détection de Stades")

        # Formules de détection de stades
        stage_formulas = [
            ("Ratio δ/θ (Sommeil Lent):", "R_{δ/θ} = P_δ / P_θ"),
            ("Ratio α/β (Éveil):", "R_{α/β} = P_α / P_β"),
            ("Complexité Spectrale:", "C = -Σ p(f) ⋅ log₂(p(f))"),
            ("Entropie de Shannon:", "H = -Σ p_i ⋅ log₂(p_i)"),
            ("Indice de Spindles:", "SI = P_{σ} / (P_δ + P_θ)"),
        ]

        for title, formula in stage_formulas:
            sub_frame = ttk.Frame(frame_stages)
            sub_frame.pack(fill=tk.X, pady=5, padx=10)

            title_label = ttk.Label(sub_frame, text=title, font=("Arial", 10, "bold"))
            title_label.pack(anchor=tk.W)

            formula_label = ttk.Label(sub_frame, text=formula, font=("Cambria Math", 11), justify=tk.CENTER)
            formula_label.pack(pady=(0, 5))

        # === Onglet 3: Filtres ===
        frame_filters = ttk.Frame(notebook)
        notebook.add(frame_filters, text="Filtres")

        # Formules de filtres
        filter_formulas = [
            ("Filtre Butterworth (Passe-bande):", "H(f) = 1 / √(1 + (f/f_c)^{2n})"),
            ("Fréquence de coupure:", "f_c = f_{low} à f_{high}"),
            ("Ordre du filtre:", "n = 4 (standard EEG)"),
            ("Filtre passe-haut:", "H(f) = 1 / √(1 + (f_c/f)^{2n})"),
            ("Filtre passe-bas:", "H(f) = 1 / √(1 + (f/f_c)^{2n})"),
        ]

        for title, formula in filter_formulas:
            sub_frame = ttk.Frame(frame_filters)
            sub_frame.pack(fill=tk.X, pady=5, padx=10)

            title_label = ttk.Label(sub_frame, text=title, font=("Arial", 10, "bold"))
            title_label.pack(anchor=tk.W)

            formula_label = ttk.Label(sub_frame, text=formula, font=("Cambria Math", 11), justify=tk.CENTER)
            formula_label.pack(pady=(0, 5))

        # === Onglet 4: Statistiques ===
        frame_stats = ttk.Frame(notebook)
        notebook.add(frame_stats, text="Statistiques")

        # Formules statistiques
        stat_formulas = [
            ("Cohérence:", "Coh(f) = |P_{xy}(f)|² / (P_x(f) ⋅ P_y(f))"),
            ("Corrélation de Pearson:", "r = Cov(x,y) / (σ_x ⋅ σ_y)"),
            ("Test de Student:", "t = (μ₁ - μ₂) / √(σ₁²/n₁ + σ₂²/n₂)"),
            ("ANOVA:", "F = MS_{between} / MS_{within}"),
        ]

        for title, formula in stat_formulas:
            sub_frame = ttk.Frame(frame_stats)
            sub_frame.pack(fill=tk.X, pady=5, padx=10)

            title_label = ttk.Label(sub_frame, text=title, font=("Arial", 10, "bold"))
            title_label.pack(anchor=tk.W)

            formula_label = ttk.Label(sub_frame, text=formula, font=("Cambria Math", 11), justify=tk.CENTER)
            formula_label.pack(pady=(0, 5))

        # Bouton de fermeture
        button_frame = ttk.Frame(window)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        close_button = ttk.Button(button_frame, text="Fermer",
                                command=window.destroy)
        close_button.pack()

        # Configuration de la scrollbar pour la molette
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Centrer la fenêtre
        window.update_idletasks()
        x = (window.winfo_screenwidth() - window.winfo_width()) // 2
        y = (window.winfo_screenheight() - window.winfo_height()) // 2
        window.geometry(f"+{x}+{y}")

        window.mainloop() if parent is None else None

    except Exception as e:
        if parent:
            messagebox.showerror("Erreur", f"Erreur lors de l'affichage des formules:\n\n{str(e)}")
        else:
            print(f"Erreur lors de l'affichage des formules: {e}")


if __name__ == "__main__":
    show_math_formulas_window()





