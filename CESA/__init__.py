"""
CCESA (Complex EEG Studio Analysis) v1.0 - Package Module
=============================================

Package modulaire CESA pour l'analyse de données EEG.
Développé pour l'Unité Neuropsychologie du Stress (IRBA).

Ce package fournit les utilitaires modulaires utilisés par
CESA (EEG Studio Analysis) v3.0 :

- esa.filters : Filtres Butterworth centralisés
- esa.scoring_io : Import scoring Excel/EDF+
- esa.theme : Thèmes UI professionnel
- esa.ui_dialogs : Dialogues d'interface spécialisés
- esa.entropy : Analyse d'entropie renormée (Issartel)
- esa.user_assistant : Assistant utilisateur et aide contextuelle

Fonctionnalités principales:
- Filtres passe-bande configurables
- Import et synchronisation de scoring
- Thèmes clair/sombre pour Tkinter
- Dialogues utilisateur spécialisés
- Analyse d'entropie renormée multi-canal
- Assistant utilisateur et aide contextuelle
- Architecture modulaire et réutilisable

Auteur: Côme Barmoy (Unité Neuropsychologie du Stress - IRBA)
Version: 3.0.0
Date: 2025-01-27
Licence: MIT
"""

__version__ = "3.0.0"
__author__ = "Côme Barmoy (IRBA)"
__all__ = ["filters", "scoring_io", "theme", "ui_dialogs", "entropy", "user_assistant"]
