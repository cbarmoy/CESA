"""
CESA v0.0beta1.1 - Advanced Theme Manager
=================================

Module de gestion de thèmes complet pour l'interface CESA (EEG Studio Analysis) v0.0beta1.1.
Développé pour l'Unité Neuropsychologie du Stress (IRBA).

Fournit un système de thèmes graphiques complet avec 3 thèmes personnalisés:
- Otilia 🦖🌸 : Thème rose avec image de coucher de soleil
- Fred 🧗‍♂️🌿 : Thème vert avec image de montagne verdoyante
- Eléna 🐢🌊 : Thème bleu avec image de plongée sous-marine

Fonctionnalités:
- Gestion centralisée des thèmes avec classe ThemeManager
- Changement de thème dynamique avec mise à jour instantanée
- Support des images de fond et palettes de couleurs cohérentes
- Configuration ttk complète et adaptative
- Support Windows/Linux avec adaptation automatique
- Couleurs optimisées pour la visualisation EEG

Utilisé par l'interface CESA pour:
- Thèmes personnalisés avec images de fond
- Interface cohérente sur tous les systèmes
- Ergonomie visuelle optimisée
- Support des utilisateurs avec préférences visuelles

Styles configurés:
- TFrame, TLabel, TButton, TEntry, TCombobox
- Treeview, Scrollbar, Notebook, Progressbar
- Tous les widgets ttk standards
- Images de fond et palettes de couleurs

Auteur: Côme Barmoy (Unité Neuropsychologie du Stress - IRBA)
Version: 0.0beta1.1
Date: 2025-09-26
"""

import tkinter as tk
from tkinter import ttk
import os
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import logging


class Theme(ABC):
    """Classe abstraite pour définir un thème graphique."""

    def __init__(self, name: str, emoji: str, primary_color: str, background_image_path: Optional[str] = None):
        self.name = name
        self.emoji = emoji
        self.primary_color = primary_color
        self.background_image_path = background_image_path
        self.stage_colors = self._define_stage_colors()
        # Palette UI étendue pour un rebranding moderne
        # Les descendants utilisent ces clés: bg, bg_alt, fg, fg_muted, accent,
        # border, surface, surface_alt, success, warning, danger
        self._ui_palette = self._define_ui_colors()

    @abstractmethod
    def _define_stage_colors(self) -> Dict[str, str]:
        """Définit les couleurs pour chaque stade de sommeil."""
        pass

    @abstractmethod
    def _define_ui_colors(self) -> Dict[str, str]:
        """Définit les couleurs pour l'interface utilisateur."""
        pass

    def get_stage_colors(self) -> Dict[str, str]:
        """Retourne les couleurs pour les stades de sommeil."""
        return self.stage_colors.copy()

    def get_ui_colors(self) -> Dict[str, str]:
        """Retourne les couleurs pour l'interface."""
        return self._ui_palette


class DefaultTheme(Theme):
    """Thème par défaut avec couleurs standard Tkinter/matplotlib."""

    def __init__(self):
        super().__init__(
            name="Default",
            emoji="🎨",
            primary_color="#4a90e2",  # Bleu standard
            background_image_path=None
        )

    def _define_stage_colors(self) -> Dict[str, str]:
        colors = {
            "W": "#1f77b4",    # Bleu matplotlib standard
            "N1": "#ff7f0e",   # Orange matplotlib standard
            "N2": "#2ca02c",   # Vert matplotlib standard
            "N3": "#d62728",   # Rouge matplotlib standard
            "R": "#9467bd",    # Violet matplotlib standard
            "U": "#8c564b"     # Brun matplotlib standard
        }
        # Ajouter des alias pour les variantes de noms
        colors.update({
            "Wake": colors["W"],
            "WAKE": colors["W"],
            "Éveil": colors["W"],
            "ÉVEIL": colors["W"],
            "Eveil": colors["W"],
            "EVEIL": colors["W"],
            "REM": colors["R"],
            "rem": colors["R"],
            "UNDEFINED": colors["U"],
            "Undefined": colors["U"],
            "undefined": colors["U"],
            "ARTIFACT": colors["U"],
            "Artifact": colors["U"],
            "artifact": colors["U"]
        })
        return colors

    def _define_ui_colors(self) -> Dict[str, str]:
        # Flat moderne (neutre clair) avec forts contrastes et surfaces
        return {
            "bg": "#f7f7fb",
            "bg_alt": "#ffffff",
            "fg": "#1f2335",
            "fg_muted": "#5b6078",
            "button_bg": "#e9ecf1",
            "button_fg": "#1f2335",
            "accent": "#4a90e2",
            "border": "#d7dbe7",
            "surface": "#ffffff",
            "surface_alt": "#f0f2f7",
            "success": "#2ecc71",
            "warning": "#f5a623",
            "danger": "#e74c3c",
        }


class OtiliaTheme(Theme):
    """Thème Otilia 🦖🌸 - Rose avec coucher de soleil."""

    def __init__(self):
        super().__init__(
            name="Otilia",
            emoji="🦖🌸",
            primary_color="#FF69B4",  # Rose
            background_image_path="CESA/backgrounds/Soleil.jpg"  # Utilise l'image de soleil
        )

    def _define_stage_colors(self) -> Dict[str, str]:
        colors = {
            "W": "#FFB6C1",    # Rose clair
            "N1": "#FF69B4",   # Rose
            "N2": "#FF1493",   # Rose foncé
            "N3": "#DC143C",   # Rouge cramoisi
            "R": "#8B008B",    # Magenta foncé
            "U": "#DDA0DD"     # Violet clair
        }
        # Ajouter des alias pour les variantes de noms
        colors.update({
            "Wake": colors["W"],
            "WAKE": colors["W"],
            "Éveil": colors["W"],
            "ÉVEIL": colors["W"],
            "Eveil": colors["W"],
            "EVEIL": colors["W"],
            "REM": colors["R"],
            "rem": colors["R"],
            "UNDEFINED": colors["U"],
            "Undefined": colors["U"],
            "undefined": colors["U"],
            "ARTIFACT": colors["U"],
            "Artifact": colors["U"],
            "artifact": colors["U"]
        })
        return colors

    def _define_ui_colors(self) -> Dict[str, str]:
        # Rose moderne flat, douceur avec contraste lisible
        return {
            "bg": "#fff5f8",
            "bg_alt": "#ffffff",
            "fg": "#5c2a4a",
            "fg_muted": "#7f4a6c",
            "button_bg": "#ffd2e2",
            "button_fg": "#5c2a4a",
            "accent": "#ff4da6",
            "border": "#f3c4d6",
            "surface": "#ffffff",
            "surface_alt": "#ffe6f0",
            "success": "#2ecc71",
            "warning": "#f5a623",
            "danger": "#e74c3c",
        }


class FredTheme(Theme):
    """Thème Fred 🧗‍♂️🌿 - Vert avec montagne verdoyante."""

    def __init__(self):
        super().__init__(
            name="Fred",
            emoji="🧗‍♂️🌿",
            primary_color="#228B22",  # Vert forêt
            background_image_path="CESA/backgrounds/Montagne.jpg"  # Utilise l'image de montagne du dossier backgrounds
        )

    def _define_stage_colors(self) -> Dict[str, str]:
        colors = {
            "W": "#90EE90",    # Vert clair
            "N1": "#32CD32",   # Vert lime
            "N2": "#228B22",   # Vert forêt
            "N3": "#006400",   # Vert foncé
            "R": "#008000",    # Vert
            "U": "#98FB98"     # Vert pâle
        }
        # Ajouter des alias pour les variantes de noms
        colors.update({
            "Wake": colors["W"],
            "WAKE": colors["W"],
            "Éveil": colors["W"],
            "ÉVEIL": colors["W"],
            "Eveil": colors["W"],
            "EVEIL": colors["W"],
            "REM": colors["R"],
            "rem": colors["R"],
            "UNDEFINED": colors["U"],
            "Undefined": colors["U"],
            "undefined": colors["U"],
            "ARTIFACT": colors["U"],
            "Artifact": colors["U"],
            "artifact": colors["U"]
        })
        return colors

    def _define_ui_colors(self) -> Dict[str, str]:
        # Vert moderne, frais mais lisible
        return {
            "bg": "#f2fbf5",
            "bg_alt": "#ffffff",
            "fg": "#114d2c",
            "fg_muted": "#2c6b45",
            "button_bg": "#d3f0dd",
            "button_fg": "#114d2c",
            "accent": "#2ecc71",
            "border": "#c3e6cf",
            "surface": "#ffffff",
            "surface_alt": "#e9f7ef",
            "success": "#2ecc71",
            "warning": "#f5a623",
            "danger": "#e74c3c",
        }


class ElenaTheme(Theme):
    """Thème Eléna 🐢🌊 - Bleu avec plongée sous-marine."""

    def __init__(self):
        super().__init__(
            name="Eléna",
            emoji="🐢🌊",
            primary_color="#1E90FF",  # Bleu
            background_image_path="CESA/backgrounds/Tortue.jpg"  # Utilise l'image de tortue du dossier backgrounds
        )

    def _define_stage_colors(self) -> Dict[str, str]:
        colors = {
            "W": "#87CEEB",    # Bleu ciel
            "N1": "#4682B4",   # Bleu acier
            "N2": "#1E90FF",   # Bleu
            "N3": "#000080",   # Bleu marine
            "R": "#4169E1",    # Bleu royal
            "U": "#9370DB"     # Violet moyen
        }
        # Ajouter des alias pour les variantes de noms
        colors.update({
            "Wake": colors["W"],
            "WAKE": colors["W"],
            "Éveil": colors["W"],
            "ÉVEIL": colors["W"],
            "Eveil": colors["W"],
            "EVEIL": colors["W"],
            "REM": colors["R"],
            "rem": colors["R"],
            "UNDEFINED": colors["U"],
            "Undefined": colors["U"],
            "undefined": colors["U"],
            "ARTIFACT": colors["U"],
            "Artifact": colors["U"],
            "artifact": colors["U"]
        })
        return colors

    def _define_ui_colors(self) -> Dict[str, str]:
        # Bleu professionnel, élégant
        return {
            "bg": "#f3f8ff",
            "bg_alt": "#ffffff",
            "fg": "#0f2747",
            "fg_muted": "#3a5172",
            "button_bg": "#d6e6ff",
            "button_fg": "#0f2747",
            "accent": "#3d7dff",
            "border": "#c7d8f0",
            "surface": "#ffffff",
            "surface_alt": "#eaf2ff",
            "success": "#2ecc71",
            "warning": "#f5a623",
            "danger": "#e74c3c",
        }


class ThemeManager:
    """
    Gestionnaire centralisé des thèmes graphiques.

    Cette classe gère:
    - La définition des thèmes disponibles
    - Le thème actuel
    - L'application des thèmes à l'interface
    - La mise à jour dynamique des couleurs
    """

    def __init__(self):
        self.themes = {
            "default": DefaultTheme(),
            "otilia": OtiliaTheme(),
            "fred": FredTheme(),
            "elena": ElenaTheme()
        }
        self.current_theme_name = "default"
        self._background_photo = None
        self._background_label = None
        self._create_default_backgrounds()

    def _create_default_backgrounds(self):
        """Crée des images de fond par défaut si elles n'existent pas."""
        try:
            # Créer le répertoire des backgrounds s'il n'existe pas
            bg_dir = "CESA/backgrounds"
            os.makedirs(bg_dir, exist_ok=True)

            # Créer des images de fond par défaut codées en base64
            backgrounds = {
                "sunset_background.jpg": self._create_sunset_background,
                "mountain_background.jpg": self._create_mountain_background,
                "underwater_background.jpg": self._create_underwater_background
            }

            for filename, create_func in backgrounds.items():
                filepath = os.path.join(bg_dir, filename)
                if not os.path.exists(filepath):
                    create_func(filepath)

        except Exception as e:
            logging.warning(f"Impossible de créer les images de fond par défaut: {e}")

    def _create_sunset_background(self, filepath: str):
        """Crée une image de coucher de soleil."""
        try:
            from PIL import Image, ImageDraw
            # Créer une image simple avec un dégradé rose/orange
            img = Image.new('RGB', (800, 600), color='#FFF0F5')
            draw = ImageDraw.Draw(img)

            # Ajouter un dégradé simple
            for y in range(600):
                r = int(255 - (y / 600) * 100)  # Dégradé vers le rose
                g = int(240 - (y / 600) * 50)
                b = int(245 - (y / 600) * 50)
                color = (r, g, b)
                draw.line([(0, y), (800, y)], fill=color)

            img.save(filepath)
        except Exception as e:
            logging.warning(f"Impossible de créer l'image de coucher de soleil: {e}")

    def _create_mountain_background(self, filepath: str):
        """Crée une image de montagne verdoyante."""
        try:
            from PIL import Image, ImageDraw
            # Créer une image simple avec un dégradé vert
            img = Image.new('RGB', (800, 600), color='#F0FFF0')
            draw = ImageDraw.Draw(img)

            # Ajouter un dégradé simple
            for y in range(600):
                r = int(240 - (y / 600) * 50)  # Dégradé vers le vert
                g = int(255 - (y / 600) * 100)
                b = int(240 - (y / 600) * 50)
                color = (r, g, b)
                draw.line([(0, y), (800, y)], fill=color)

            img.save(filepath)
        except Exception as e:
            logging.warning(f"Impossible de créer l'image de montagne: {e}")

    def _create_underwater_background(self, filepath: str):
        """Crée une image de plongée sous-marine."""
        try:
            from PIL import Image, ImageDraw
            # Créer une image simple avec un dégradé bleu
            img = Image.new('RGB', (800, 600), color='#F0F8FF')
            draw = ImageDraw.Draw(img)

            # Ajouter un dégradé simple
            for y in range(600):
                r = int(240 - (y / 600) * 50)  # Dégradé vers le bleu
                g = int(248 - (y / 600) * 100)
                b = int(255 - (y / 600) * 150)
                color = (r, g, b)
                draw.line([(0, y), (800, y)], fill=color)

            img.save(filepath)
        except Exception as e:
            logging.warning(f"Impossible de créer l'image sous-marine: {e}")

    def get_current_theme(self) -> Theme:
        """Retourne le thème actuel."""
        return self.themes[self.current_theme_name]

    def set_theme(self, theme_name: str):
        """Change le thème actuel."""
        if theme_name in self.themes:
            self.current_theme_name = theme_name
            logging.info(f"🎨 Thème changé: {self.get_current_theme().name} {self.get_current_theme().emoji}")

    def get_available_themes(self) -> Dict[str, str]:
        """Retourne la liste des thèmes disponibles."""
        return {name: f"{theme.name} {theme.emoji}" for name, theme in self.themes.items()}

    def apply_theme_to_widget(self, widget, theme: Optional[Theme] = None):
        """Applique un thème à un widget."""
        if theme is None:
            theme = self.get_current_theme()

        try:
            ui_colors = theme.get_ui_colors()

            # Pour les widgets ttk - utiliser le système de style
            if isinstance(widget, ttk.Widget):
                # Créer un style personnalisé pour ce thème
                style_name = f"Custom{theme.name}.T{widget.winfo_class()}"

                # Configurer le style
                style = ttk.Style()
                try:
                    style.configure(style_name,
                                  background=ui_colors.get('bg', '#FFFFFF'),
                                  foreground=ui_colors.get('fg', '#000000'))
                    # Appliquer le style au widget
                    widget.configure(style=style_name)
                except Exception as e:
                    # Fallback: essayer sans style personnalisé
                    logging.debug(f"Impossible de configurer le style {style_name}: {e}")
                    try:
                        widget.configure(style='')  # Remettre le style par défaut
                    except:
                        pass

            # Pour les widgets Tkinter natifs - essayer de changer les couleurs
            elif hasattr(widget, 'configure'):
                try:
                    widget.configure(bg=ui_colors.get('bg', '#FFFFFF'),
                                   fg=ui_colors.get('fg', '#000000'))
                except:
                    # Certains widgets natifs ignorent les couleurs sous Windows/macOS
                    pass

        except Exception as e:
            logging.warning(f"Erreur application thème au widget: {e}")

    def apply_theme_to_root(self, root, theme: Optional[Theme] = None):
        """Applique un thème à la fenêtre principale."""
        if theme is None:
            theme = self.get_current_theme()

        try:
            ui_colors = theme.get_ui_colors()

            # Appliquer les couleurs de base à la racine
            root.configure(bg=ui_colors.get('bg', '#FFFFFF'))

            # Configurer le style ttk
            style = ttk.Style(root)
            style.theme_use('clam')

            # Définir les styles pour ce thème avec des couleurs plus contrastées
            button_bg = ui_colors.get('button_bg', '#CCCCCC')
            button_fg = ui_colors.get('button_fg', '#000000')

            # S'assurer que les couleurs sont suffisamment contrastées
            if button_bg == ui_colors.get('bg', '#FFFFFF'):
                # Si le fond du bouton est le même que le fond, utiliser une couleur légèrement différente
                button_bg = self._adjust_color(button_bg, -10)  # Plus sombre

            # Palette générale
            style.configure('Custom.TFrame', background=ui_colors.get('bg', '#FFFFFF'))
            style.configure('Custom.TLabelframe', background=ui_colors.get('surface', '#FFFFFF'),
                           bordercolor=ui_colors.get('border', '#DDDDDD'), relief='solid', borderwidth=1)
            style.configure('Custom.TLabel', background=ui_colors.get('bg', '#FFFFFF'),
                          foreground=ui_colors.get('fg', '#000000'))
            style.configure('Custom.TButton', background=button_bg, foreground=button_fg,
                          relief='flat', borderwidth=1)
            # Micro-animations/hover: léger foncé et appui visuel
            style.map('Custom.TButton',
                      background=[('active', self._adjust_color(button_bg, -8)), ('pressed', self._adjust_color(button_bg, -14))],
                      relief=[('pressed', 'sunken')])
            try:
                root.option_add('*TButton.cursor', 'hand2')
            except Exception:
                pass
            # Styles modernes additionnels pour toolbar/contrôles
            style.configure('Modern.TFrame', background=ui_colors.get('surface', '#FFFFFF'))
            style.configure('Modern.TButton', background=ui_colors.get('accent', '#4a90e2'),
                            foreground='#ffffff', relief='flat', borderwidth=0, padding=6)
            style.map('Modern.TButton',
                      background=[('active', self._adjust_color(ui_colors.get('accent', '#4a90e2'), -10))])

            # Notebook (onglets)
            style.configure('TNotebook', background=ui_colors.get('bg', '#FFFFFF'), borderwidth=0)
            style.configure('TNotebook.Tab', background=ui_colors.get('surface', '#FFFFFF'),
                            foreground=ui_colors.get('fg', '#000000'), padding=(10, 6))
            style.map('TNotebook.Tab',
                      background=[('selected', ui_colors.get('surface_alt', '#F0F2F7'))],
                      foreground=[('selected', ui_colors.get('fg', '#000000'))])
            
            style.configure('Custom.TCheckbutton', background=ui_colors.get('bg', '#FFFFFF'),
                          foreground=ui_colors.get('fg', '#000000'))
            style.configure('Custom.TScale', background=ui_colors.get('bg', '#FFFFFF'))
            style.configure('Custom.Horizontal.TScale', troughcolor=ui_colors.get('surface_alt', '#EEEEEE'),
                            background=ui_colors.get('accent', '#4a90e2'))

            # Appliquer l'image de fond si disponible
            self._apply_background_image(root, theme)

            # Fonction récursive pour appliquer le thème à tous les widgets
            def _apply_to_children(widget):
                try:
                    # Appliquer le thème au widget actuel
                    self.apply_theme_to_widget(widget, theme)

                    # Appliquer aux enfants
                    for child in widget.winfo_children():
                        _apply_to_children(child)
                except Exception as e:
                    logging.debug(f"Erreur application thème à widget: {e}")

            _apply_to_children(root)

        except Exception as e:
            logging.warning(f"Erreur application thème à root: {e}")

    def _apply_background_image(self, root, theme: Theme):
        """Applique l'image de fond du thème (utilise un seul widget pour éviter la superposition)."""
        try:
            # Supprimer l'ancien label d'arrière-plan s'il existe
            if self._background_label is not None:
                try:
                    self._background_label.destroy()
                except:
                    pass
                self._background_label = None

            if theme.background_image_path:
                # Le chemin peut être relatif ou absolu
                bg_path = theme.background_image_path

                # Si c'est un chemin relatif, le résoudre par rapport au répertoire de travail
                if not os.path.isabs(bg_path):
                    bg_path = os.path.join(os.getcwd(), bg_path)

                if os.path.exists(bg_path):
                    # Créer un PhotoImage pour l'arrière-plan
                    self._background_photo = tk.PhotoImage(file=bg_path)

                    # Créer un label pour afficher l'arrière-plan (un seul)
                    self._background_label = tk.Label(root, image=self._background_photo)
                    self._background_label.place(x=0, y=0, relwidth=1, relheight=1)
                    self._background_label.lower()  # Mettre en arrière-plan
                    logging.info(f"✅ Image de fond appliquée: {bg_path}")
                else:
                    logging.warning(f"⚠️ Image de fond introuvable: {bg_path}")

        except Exception as e:
            logging.debug(f"Impossible d'appliquer l'image de fond: {e}")

    def apply_background_to_figure(self, fig, theme: Optional[Theme] = None):
        """Applique l'image de fond du thème à une figure matplotlib."""
        try:
            if theme is None:
                theme = self.get_current_theme()

            ui_colors = theme.get_ui_colors()

            # Changer le fond de la figure avec les couleurs du thème
            fig.patch.set_facecolor(ui_colors.get('bg', '#ffffff'))

            # Supprimer les anciens axes d'arrière-plan s'ils existent
            for ax in fig.axes:
                if hasattr(ax, '_theme_background'):
                    ax.remove()
                    break

            # Ajouter l'image de fond si disponible
            if theme.background_image_path:
                # Le chemin peut être relatif ou absolu
                bg_path = theme.background_image_path

                # Si c'est un chemin relatif, le résoudre par rapport au répertoire de travail
                if not os.path.isabs(bg_path):
                    bg_path = os.path.join(os.getcwd(), bg_path)

                if os.path.exists(bg_path):
                    # Charger l'image avec PIL
                    from PIL import Image
                    img = Image.open(bg_path)

                    # Créer un axe pour l'image de fond (un seul)
                    ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
                    ax_bg.imshow(img, aspect='auto', extent=[0, 1, 0, 1], alpha=0.1)
                    ax_bg.axis('off')  # Cacher les axes
                    ax_bg._theme_background = True  # Marquer comme axe d'arrière-plan

                    logging.info(f"✅ Image de fond appliquée à la figure: {bg_path}")

            # Forcer le rafraîchissement de la figure
            fig.canvas.draw_idle()

        except Exception as e:
            logging.debug(f"Impossible d'appliquer l'image de fond à la figure: {e}")

    def _adjust_color(self, hex_color: str, adjustment: int) -> str:
        """Ajuste une couleur hexadécimale en ajoutant une valeur à chaque composante RGB."""
        try:
            # Convertir hex en RGB
            hex_color = hex_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)

            # Ajuster
            r = max(0, min(255, r + adjustment))
            g = max(0, min(255, g + adjustment))
            b = max(0, min(255, b + adjustment))

            # Reconvertir en hex
            return f"#{r:02x}{g:02x}{b:02x}"
        except:
            return hex_color

    def get_stage_colors(self) -> Dict[str, str]:
        """Retourne les couleurs des stades pour le thème actuel."""
        return self.get_current_theme().get_stage_colors()


# Instance globale du gestionnaire de thèmes
theme_manager = ThemeManager()
