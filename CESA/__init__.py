"""
CESA (Complex EEG Studio Analysis) - Package Module
====================================================

Package modulaire CESA pour l'analyse de données EEG/PSG.
Développé pour l'Unité Neuropsychologie du Stress (IRBA).

Ce package fournit les utilitaires modulaires utilisés par CESA :

- filters / filter_engine : Filtrage composable (Bandpass, Notch, etc.)
- scoring_io : Import scoring Excel/EDF+
- sleep_pipeline : Scoring AASM avec ML/DL et HMM
- theme : Thèmes UI professionnel (dark/light)
- ui_dialogs : Dialogues d'interface spécialisés
- entropy : Analyse d'entropie renormée (Issartel)
- report_generator : Rapports HTML/PDF intégrés
- user_assistant : Assistant utilisateur et aide contextuelle

Auteur: Côme Barmoy (Unité Neuropsychologie du Stress - IRBA)
Version: 0.0beta1.0
Date: 2026-04-05
Licence: MIT
"""

__version__ = "0.0beta1.0"
__author__ = "Côme Barmoy (IRBA)"
__all__ = [
    "filters", "filter_engine", "scoring_io", "theme", "ui_dialogs",
    "entropy", "user_assistant", "group_analysis", "report_generator",
]
