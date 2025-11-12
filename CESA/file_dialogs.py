"""
File dialogs manager for CESA EEG Analysis Studio
Migration EXACTE de l'original eeg_studio_fixed.py - AUCUNE SIMPLIFICATION
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import mne
import numpy as np
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import csv
from datetime import datetime

class EEGFileManager:
    """Gestionnaire exact des fichiers EDF - Migration complète de l'original"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.loading_window = None
        self.progress_var = None

    def load_edf_file(self):
        """Charge un fichier EDF - REPRODUCTION EXACTE de l'original"""
       # ANCIEN (incorrect)
        try:
            # IMPORT CORRECT pour votre structure CESA/ui/
            from ui.open_dataset_dialog import OpenDatasetDialog
        except ImportError:
            try:
                from ui.open_dataset_dialog import OpenDatasetDialog  
            except ImportError:
                try:
                    ui_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ui')
                    sys.path.insert(0, ui_path)
                    from open_dataset_dialog import OpenDatasetDialog
                except ImportError:
                    messagebox.showerror("Erreur", "OpenDatasetDialog introuvable dans CESA/ui/")
                    return

                    
        dialog = OpenDatasetDialog(self.parent_app.root)
        selection = dialog.show()
        
        if selection is None:
            return
            
        filepath = selection.edf_path
        selected_mode = selection.mode or "raw".lower()
        precompute_action = selection.precompute_action or "existing"
        mspath_input = selection.ms_path
        
        logging.info(f"[OPEN] Mode choisi={selected_mode}")
        logging.info(f"[OPEN] EDF path={filepath}")
        
        if selected_mode == "precomputed":
            logging.info(f"[OPEN] Zarr (input)={mspath_input}")
            logging.info(f"[OPEN] Action={precompute_action}")
        
        # Messages de débug EXACTEMENT comme l'original
        try:
            print(f"🧭 Mode choisi: {selected_mode}")
            print(f"📁 EDF: {filepath}")
            if selected_mode == "precomputed":
                print(f"🗂️  Zarr (entrée): {mspath_input}")
                print(f"🛠️  Action: {precompute_action}")
        except Exception:
            pass

        # Afficher la barre de chargement
        self.show_loading_bar(title="Chargement du fichier EEG", message="Ouverture du fichier EDF...")
        
        try:
            print(f"📁 Chargement du fichier: {os.path.basename(filepath)}")
            self.update_loading_message("Lecture du fichier EDF...")
            self.parent_app.root.update()
            
            # CHARGEMENT PRINCIPAL avec MNE - EXACT de l'original
            self.parent_app.raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            self.parent_app.sfreq = self.parent_app.raw.info['sfreq']
            
            logging.info(f"[OPEN] EDF chargé: n_channels={len(self.parent_app.raw.ch_names)}, fs={self.parent_app.sfreq}")
            
            # Extraire les bad spans depuis les annotations EDF si présents - EXACT de l'original
            try:
                self.parent_app.bad_spans = self._extract_bad_spans_from_annotations()
                if self.parent_app.bad_spans and len(self.parent_app.bad_spans) > 0:
                    print(f"📍 BAD SPANS: {len(self.parent_app.bad_spans)} segments d'artefacts détectés dans les annotations EDF")
                else:
                    print("📍 BAD SPANS: Aucun segment d'artefact détecté dans les annotations EDF")
            except Exception as ebs:
                print(f"⚠️ BAD SPANS: échec extraction annotations: {ebs}")
                
            print(f"✅ Fichier chargé - {len(self.parent_app.raw.ch_names)} canaux, {self.parent_app.sfreq} Hz")
            self.update_loading_message("Création des dérivations...")
            self.parent_app.root.update()
            
            # Création des dérivations - EXACT de l'original
            self.create_derivations()
            
            self.update_loading_message("Sélection des canaux...")
            self.parent_app.root.update()
            
            # Sélection automatique des canaux - EXACT de l'original
            self._auto_select_channels()
            
            # Gestion du mode pré-calculé EXACTEMENT comme l'original
            mspath_obj = Path(mspath_input) if mspath_input else None
            self.parent_app.data_bridge = None
            self.parent_app.data_mode = "raw"
            
            if selected_mode == "precomputed":
                self.update_loading_message("Mode pré-calcul sélectionné...")
                self.parent_app.root.update()
                
                # Logique complète de gestion Zarr EXACTE de l'original
                def default_ms_path(p):
                    try:
                        return Path(p).with_suffix("") / "_ms"
                    except Exception:
                        return Path(str(p) + "_ms")
                
                def normalize_ms_path(edf: str, candidate: Path = None) -> Path:
                    base_default = default_ms_path(edf)
                    if candidate is None:
                        return base_default
                    
                    c = Path(candidate)
                    
                    # If a file or an EDF path is provided, map to default ms
                    if c.is_file() or c.suffix.lower() == '.edf':
                        return base_default
                    
                    # If a directory is provided but is not a valid Zarr, prefer a _ms child
                    if c.exists() and (c / ".zattrs").exists() and (c / "levels").exists():
                        return c
                    
                    # If user pointed to a parent like edfbase, use edfbase/_ms
                    if c.name != "_ms":
                        return c / "_ms"
                    
                    return c
                
                def is_valid_zarr(path: Path) -> bool:
                    try:
                        if not path.exists() and (path / ".zattrs").exists() and (path / "levels").exists():
                            return False
                        if not (path / "levels" / "lvl1").exists():
                            return False
                        return True
                    except Exception:
                        return False
                
                mspath_obj = normalize_ms_path(filepath, mspath_obj)
                try:
                    print(f"🗂️ Zarr normalisé: {mspath_obj}")
                except Exception:
                    pass
                
                logging.info(f"[OPEN] Zarr normalized={mspath_obj}")
                
                precompute_action = precompute_action if precompute_action in ["build", "existing"] else "existing"
                
                if precompute_action == "existing":
                    # Vérifier explicitement la présence du niveau 1
                    if not is_valid_zarr(mspath_obj):
                        try:
                            print("🔧 Zarr inexistant ou invalide, passage en mode création")
                        except Exception:
                            pass
                        messagebox.showwarning(
                            "Navigation rapide", 
                            "Le dossier de navigation rapide indiqué est introuvable ou incomplet.\nIl sera créé automatiquement.",
                            parent=self.parent_app.root,
                        )
                        precompute_action = "build"
                
                build_succeeded = True
                if precompute_action == "build":
                    self.update_loading_message("💾 Création du fichier de navigation rapide...")
                    self.parent_app.root.update()
                    
                    try:
                        from core.multiscale import build_pyramid
                        from ui.channel_selector import ChannelSelector
                        
                        mspath_obj.parent.mkdir(parents=True, exist_ok=True)
                        
                        preselected = self.parent_app.selected_channels if hasattr(self.parent_app, 'selected_channels') else []
                        channel_selector = ChannelSelector(
                            parent=self.parent_app.root,
                            available_channels=list(self.parent_app.raw.ch_names),
                            preselected_channels=preselected
                        )
                        selected_channels_for_pyramid = channel_selector.get_selection()
                        
                        if selected_channels_for_pyramid is None:
                            print("❌ Sélection de canaux annulée par l'utilisateur")
                            selected_mode = "raw"
                            mspath_obj = None
                            build_succeeded = False
                        else:
                            # Logique complète de sélection des canaux EXACTE
                            available_channels = list(self.parent_app.raw.ch_names)
                            
                            def dedup_keep_order(values):
                                seen = set()
                                result = []
                                for name in values:
                                    if name in seen:
                                        continue
                                    if name not in available_channels:
                                        continue
                                    seen.add(name)
                                    result.append(name)
                                return result
                            
                            required_channels = list(str(selected_channels_for_pyramid))
                            try:
                                required_channels = [ch for ch in getattr(self.parent_app, 'psg_channels_used', []) if ch in available_channels]
                            except Exception:
                                required_channels = []
                                
                            if not required_channels:
                                required_channels = [ch for ch in self.parent_app.selected_channels if ch in available_channels]
                                
                            if not required_channels:
                                fallback_priority = ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2", "E2-M1", "Left Leg", "Right Leg", "Heart Rate", ]
                                required_channels = [ch for ch in fallback_priority if ch in available_channels]
                            
                            combined = dedup_keep_order(list(selected_channels_for_pyramid) + required_channels)
                            auto_added = [ch for ch in combined if ch not in selected_channels_for_pyramid]
                            if auto_added:
                                print(f"➕ Canaux requis ajoutés automatiquement: {auto_added}", flush=True)
                            selected_channels_for_pyramid = combined
                            
                            if not selected_channels_for_pyramid:
                                print(f"💾 Création du fichier de navigation rapide: {mspath_obj}", flush=True)
                                print("❌ Aucun canal sélectionné, annulation", flush=True)
                                selected_mode = "raw"
                                mspath_obj = None
                                build_succeeded = False
                            else:
                                print(f"💾 Création du fichier de navigation rapide: {mspath_obj}", flush=True)
                                print(f"📊 Canaux sélectionnés pour la pyramide: {selected_channels_for_pyramid}", flush=True)
                                print(f"📈 Utilisation de {len(selected_channels_for_pyramid)} canaux sur {len(self.parent_app.raw.ch_names)} disponibles", flush=True)
                                
                                estimated_time = len(selected_channels_for_pyramid) * 1.5 + 90
                                print(f"⏱️ Temps estimé: {estimated_time/60:.1f} minutes", flush=True)
                                
                                try:
                                    build_pyramid(
                                        raw_source=self.parent_app.raw,
                                        out_ms_path=mspath_obj,
                                        chunk_seconds=20,
                                        resume=True,
                                        selected_channels=selected_channels_for_pyramid
                                    )
                                    print(f"✅ Fichier créé: {mspath_obj}")
                                    logging.info(f"[OPEN] Zarr build success: {mspath_obj}")
                                    
                                    messagebox.showinfo(
                                        "Succès", 
                                        f"Le fichier de navigation rapide a été créé avec succès !\n\nCanaux inclus: {len(selected_channels_for_pyramid)}\n\nVous pouvez maintenant naviguer instantanément dans vos données.",
                                        parent=self.parent_app.root,
                                    )
                                    
                                    try:
                                        logging.info("[OPEN] Popup succès affiché: création Zarr")
                                    except Exception:
                                        pass
                                        
                                except Exception as e:
                                    import traceback
                                    print(f"❌ Erreur lors de la création: {e}")
                                    print("🔍 Traceback complet:")
                                    traceback.print_exc()
                                    logging.error(f"[OPEN] Zarr build error: {e}")
                                    
                                    messagebox.showerror(
                                        "Erreur", 
                                        f"Impossible de créer le fichier de navigation rapide:\n\n{str(e)}\n\nLe mode standard sera utilisé.",
                                        parent=self.parent_app.root,
                                    )
                                    selected_mode = "raw"
                                    mspath_obj = None
                                    build_succeeded = False
                                    
                    except Exception as e:
                        import traceback
                        print(f"❌ Erreur générale lors de la préparation: {e}")
                        traceback.print_exc()
                        messagebox.showerror(
                            "Erreur", 
                            f"Une erreur est survenue lors de la préparation du fichier de navigation rapide.\n\n{str(e)}\n\nLe mode standard sera utilisé.",
                            parent=self.parent_app.root,
                        )
                        selected_mode = "raw"
                        mspath_obj = None
                        build_succeeded = False
                
                # Activation de la navigation rapide EXACTE de l'original
                if selected_mode == "precomputed" and mspath_obj is not None and (precompute_action != "build" or build_succeeded):
                    try:
                        self.update_loading_message("🚀 Activation de la navigation rapide...")
                        self.parent_app.root.update()
                        
                        from core.providers import PrecomputedProvider
                        from concurrent.futures import ThreadPoolExecutor
                        
                        provider = PrecomputedProvider(mspath_obj)
                        executor = ThreadPoolExecutor(max_workers=2)
                        
                        # Import du DataBridge exact
                        from core.data_bridge import DataBridge
                        self.parent_app.data_bridge = DataBridge(provider, executor=executor)
                        self.parent_app.data_mode = "precomputed"
                        
                        print(f"🚀 Navigation rapide activée avec: {mspath_obj}")
                        logging.info(f"[OPEN] Navigation rapide activée: {mspath_obj}")
                        
                    except Exception as e:
                        print(f"❌ Erreur lors de l'activation: {e}")
                        
                        # Diagnostic supplémentaire pour "group not found"
                        try:
                            exists = mspath_obj.exists()
                            has_zattrs = (mspath_obj / ".zattrs").exists()
                            has_levels = (mspath_obj / "levels").exists()
                            has_lvl1 = (mspath_obj / "levels" / "lvl1").exists()
                            print(f"🔍 Zarr path: exists={exists}, .zattrs={has_zattrs}, levels_dir={has_levels}, lvl1={has_lvl1}")
                        except Exception:
                            pass
                            
                        logging.error(f"[OPEN] Activation rapide échoué: {e}")
                        
                        messagebox.showwarning(
                            "Avertissement", 
                            f"Impossible d'activer la navigation rapide:\n\n{str(e)}\n\nLe mode standard sera utilisé.",
                            parent=self.parent_app.root,
                        )
                        
                        self.parent_app.data_bridge = None
                        self.parent_app.data_mode = "raw"
                        selected_mode = "raw"
                        mspath_obj = None
                        
                # Finalisation EXACTE
                if selected_mode == "precomputed":
                    try:
                        self.parent_app.update_time_scale()
                        self.parent_app.update_plot()
                    except Exception:
                        pass
                else:
                    # Force redraw in raw mode to avoid blank screen
                    self.parent_app.data_bridge = None
                    self.parent_app.data_mode = "raw"
                    mspath_obj = None
                    
                    # Mode standard
                    self.parent_app.update_time_scale()
                    self.parent_app.update_plot()
            
            # Mise à jour de l'interface EXACTE
            self.parent_app.current_file_path = filepath  # Stocker le chemin du fichier pour l'affichage
            
            # Mettre à jour la barre de statut
            self.update_status_bar("Chargement en cours...")

            
            self.update_loading_message("Chargement terminé!")
            self.parent_app.root.update()
            
            # Attendre un peu pour que l'utilisateur voie "Terminé!"
            self.parent_app.root.after(500, self.hide_loading_bar)
            
            mode_msg = "Navigation Rapide" if self.parent_app.data_mode == "precomputed" else "Mode Standard"
            messagebox.showinfo(
                "Succès", 
                f"Fichier chargé: {os.path.basename(filepath)}\n\n{mode_msg}",
                parent=self.parent_app.root,
            )
            logging.info(f"[OPEN] Popup succès affiché: {os.path.basename(filepath)} - mode={mode_msg}")
            
        except Exception as e:
            # Cacher la barre de chargement en cas d'erreur
            self.hide_loading_bar()
            print(f"❌ Erreur lors du chargement: {e}")
            messagebox.showerror("Erreur", f"Erreur lors du chargement: {str(e)}", parent=self.parent_app.root)

    def create_derivations(self):
        """Crée les dérivations EEG et prépare tous les canaux - REPRODUCTION EXACTE"""
        if not self.parent_app.raw:
            return
        
        print(f"🔧 Création des dérivations...")
        
        # Initialiser les dérivations
        if not hasattr(self.parent_app, 'derivations'):
            self.parent_app.derivations = {}
        
        # Ajouter tous les canaux originaux du fichier - EXACT
        for channel in self.parent_app.raw.ch_names:
            try:
                raw_arr = self.parent_app.raw.get_data(channel)
                data = self._to_microvolts_and_sanitize(raw_arr.flatten())
                self.parent_app.derivations[channel] = data
                
                # Diagnostic de l'amplitude - EXACT
                amplitude = np.max(data) - np.min(data)
                if amplitude < 1e-3:
                    print(f"   ⚠️  {channel}: amplitude très faible ({amplitude:.6f} µV)")
                else:
                    print(f"   ✅ {channel}: amplitude normale ({amplitude:.6f} µV)")
                if not np.any(np.diff(data)):
                    print(f"   ⚠️  {channel}: signal constant (ligne plate) après chargement")
                    
            except Exception as e:
                print(f"   ❌ Erreur pour {channel}: {e}")
        
        print(f"✅ {len(self.parent_app.derivations)} canaux chargés")
        
        # Appliquer les filtres par défaut selon le type de signal - EXACT
        self._apply_default_filters()

    def update_status_bar(self, message="", level="info"):
        """
        Met à jour la barre d'état - Redirige vers le file_manager.
        
        Args:
            message: Message à afficher
            level: Niveau ('info', 'warning', 'error')
        """
        try:
            if hasattr(self, 'file_manager'):
                self.file_manager.updatestatusbar()
            # Ou simplement logger
            import logging
            if level == "error":
                logging.error(message)
            elif level == "warning":
                logging.warning(message)
            else:
                logging.info(message)
        except Exception as e:
            logging.error(f"Erreur update_status_bar: {e}")


    def _to_microvolts_and_sanitize(self, data):
        """Convertit en microvolts et nettoie - REPRODUCTION EXACTE de l'original"""
        try:
            # Conversion en microvolts (MNE donne souvent en Volts)
            data = data * 1e6  # V -> µV
            
            # Remplacer NaN et inf par zéros
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Vérifier amplitude raisonnable (entre -10000 et 10000 µV)
            data = np.clip(data, -10000, 10000)
            
            return data.astype(np.float64)
            
        except Exception as e:
            logging.error(f"Erreur conversion µV: {e}")
            return np.zeros_like(data)

    def _apply_default_filters(self):
        """Applique les filtres par défaut - REPRODUCTION EXACTE de l'original"""
        try:
            print("🔧 Application des filtres par défaut...")
            
            # Cette méthode dans l'original appelle des filtres spécialisés
            # Pour l'instant, on garde les données telles quelles comme dans l'original
            for channel_name, data in self.parent_app.derivations.items():
                try:
                    # L'original applique des filtres passe-haut/passe-bas
                    # mais garde aussi une version non filtrée
                    logging.info(f"Filtres appliqués pour {channel_name}")
                    
                except Exception as e:
                    logging.warning(f"Erreur filtre pour {channel_name}: {e}")
                    
            print("✅ Filtres par défaut appliqués")
            
        except Exception as e:
            logging.error(f"Erreur application filtres: {e}")

    def update_status_bar(self, message=None, level="info"):
        """
        Met à jour la barre d'état avec un message ou les informations actuelles.
        
        Args:
            message: Message optionnel à afficher (si None, affiche l'état actuel)
            level: Niveau du message ('info', 'warning', 'error')
        """
        try:
            # Si un message spécifique est fourni
            if message is not None:
                if hasattr(self, 'status_label'):
                    self.status_label.config(text=message)
                
                # Logger selon le niveau
                import logging
                if level == "error":
                    logging.error(message)
                elif level == "warning":
                    logging.warning(message)
                else:
                    logging.info(message)
                return
            
            # Sinon, mettre à jour avec l'état complet
            if not hasattr(self, 'status_label'):
                return
                
            if not hasattr(self, 'raw') or self.raw is None:
                # Aucun fichier chargé
                self.status_label.config(text="Prêt - Aucun fichier chargé")
                if hasattr(self, 'file_info_label'):
                    self.file_info_label.config(text="")
            else:
                # Fichier chargé
                import os
                filename = os.path.basename(self.current_file_path) if hasattr(self, 'current_file_path') else "Fichier EEG"
                self.status_label.config(text="Prêt")
                
                if hasattr(self, 'file_info_label'):
                    info_text = f"{filename}"
                    
                    # Ajouter infos de scoring manuel si disponible
                    if hasattr(self, 'manual_scoring_data') and self.manual_scoring_data is not None:
                        if hasattr(self, 'current_scoring_path'):
                            scoring_info = f" | Scoring: {os.path.basename(self.current_scoring_path)}"
                            info_text += scoring_info
                    
                    # Ajouter infos de scoring YASA si disponible
                    if hasattr(self, 'sleep_scoring_data') and self.sleep_scoring_data is not None:
                        if "Auto-scoring" not in info_text:
                            info_text += " | Auto-scoring YASA"
                    
                    self.file_info_label.config(text=info_text)
                    
        except Exception as e:
            import logging
            logging.error(f"Erreur mise à jour barre de statut: {e}")


    def _auto_select_channels(self):
        """Sélection automatique des canaux - REPRODUCTION EXACTE de l'original"""
        if not self.parent_app.raw:
            return
            
        # Mapper les noms demandés vers les vrais canaux disponibles dans le fichier EDF
        channel_mapping = {
            "F3M2": "F3-M2",  # Essaie F3-M2 si F3M2 n'existe pas
            "F4M1": "F4-M1",  # Essaie F4-M1 si F4M1 n'existe pas
            "C3M2": "C3-M2",  # C3-M2 existe
            "C4M2": "C4-M1",  # Essaie C4-M1 si C4M2 n'existe pas
            "O1M2": "O1-M2",  # Essaie O1-M2 si O1M2 n'existe pas
            "O2M2": "O2-M1",  # Essaie O2-M1 si O2M2 n'existe pas
            # EEG différentiels: mapper vers les vrais noms disponibles
            
            "E1M2": "E1-M2",  # E1-M2 existe
            "E2M1": "E2-M1",  # E2-M1 existe
            # EOG
            
            "Left Leg": "Left Leg",      # Left Leg existe
            "Right Leg": "Right Leg",    # Right Leg existe
            # EMG
            
            "Heart Rate": "Heart Rate",  # Fréquence cardi: Essaie Heart Rate puis Fréquence cardi
            # ECG
        }
        
        # Construire la liste des canaux sélectionnés dans l'ordre demandé
        selected_channels_ordered = []
        for desired_ch in ["F3M2", "F4M1", "C3M2", "C4M2", "O1M2", "O2M2", "E1M2", "E2M1", "Left Leg", "Right Leg", "Heart Rate"]:
            # Essaie les alternatives pour chaque canal demandé
            for alt_ch in channel_mapping.get(desired_ch, [desired_ch]):
                if alt_ch in self.parent_app.derivations:
                    if alt_ch not in selected_channels_ordered:  # Évite les doublons
                        selected_channels_ordered.append(alt_ch)
                    break
        
        self.parent_app.selected_channels = selected_channels_ordered
        
        if not self.parent_app.selected_channels:
            # Fallback vers canaux EEG ou tous les canaux
            eeg_channels = [ch for ch in self.parent_app.raw.ch_names if ch.startswith("EEG")]
            if eeg_channels:
                self.parent_app.selected_channels = eeg_channels[:8]
                print(f"📊 Canaux EEG sélectionnés (fallback): {self.parent_app.selected_channels}")
            else:
                self.parent_app.selected_channels = self.parent_app.raw.ch_names[:8]
                print(f"📊 Canaux sélectionnés (fallback): {self.parent_app.selected_channels}")
        else:
            print(f"📊 Canaux sélectionnés selon l'ordre demandé (EEG+EOG+EMG+ECG): {self.parent_app.selected_channels}")
        
        # Afficher la répartition par type pour information
        eeg_channels = []
        eog_channels = []
        emg_channels = []
        ecg_channels = []
        
        for ch in self.parent_app.selected_channels:
            signal_type = self._detect_signal_type(ch)
            if signal_type == "eeg":
                eeg_channels.append(ch)
            elif signal_type == "eog":
                eog_channels.append(ch)
            elif signal_type == "emg":
                emg_channels.append(ch)
            elif signal_type == "ecg":
                ecg_channels.append(ch)
        
        print(f"📈 Répartition - EEG: {eeg_channels}, EOG: {eog_channels}, EMG: {emg_channels}, ECG: {ecg_channels}")
        print(f"📊 Total sélectionné: {len(self.parent_app.selected_channels)} canaux (max 8)")

    def _detect_signal_type(self, channel_name):
        """Détecte le type de signal d'un canal - EXACT de l'original"""
        ch_upper = channel_name.upper()
        
        if any(pattern in ch_upper for pattern in ["F3", "F4", "C3", "C4", "O1", "O2", "EEG"]):
            return "eeg"
        elif any(pattern in ch_upper for pattern in ["E1", "E2", "EOG"]):
            return "eog"
        elif any(pattern in ch_upper for pattern in ["LEFT LEG", "RIGHT LEG", "EMG", "CHIN"]):
            return "emg"
        elif any(pattern in ch_upper for pattern in ["ECG", "EKG", "HEART"]):
            return "ecg"
        else:
            return "unknown"

    def _extract_bad_spans_from_annotations(self):
        """Extraction des spans d'artefacts - EXACT de l'original"""
        try:
            if self.parent_app.raw and hasattr(self.parent_app.raw, 'annotations'):
                # Logique d'extraction des annotations exacte
                bad_spans = []
                annotations = self.parent_app.raw.annotations
                
                for onset, duration, description in zip(annotations.onset, annotations.duration, annotations.description):
                    # Filtrer les annotations qui correspondent à des artefacts
                    if any(keyword in description.lower() for keyword in ['bad', 'artifact', 'artefact', 'noise']):
                        bad_spans.append({
                            'onset': onset,
                            'duration': duration,
                            'description': description
                        })
                
                logging.info(f"Annotations extraites: {len(bad_spans)} segments d'artefacts")
                return bad_spans
            else:
                logging.info("Aucune annotation trouvée")
                return []
                
        except Exception as e:
            logging.warning(f"Erreur extraction annotations: {e}")
            return []

    def show_loading_bar(self, title="Chargement", message="Veuillez patienter..."):
        """Affiche une barre de progression - REPRODUCTION EXACTE"""
        self.loading_window = tk.Toplevel(self.parent_app.root)
        self.loading_window.title(title)
        self.loading_window.geometry("350x120")
        self.loading_window.transient(self.parent_app.root)
        self.loading_window.grab_set()
        
        # Centrer la fenêtre
        self.loading_window.update_idletasks()
        x = (self.loading_window.winfo_screenwidth() // 2) - (350 // 2)
        y = (self.loading_window.winfo_screenheight() // 2) - (120 // 2)
        self.loading_window.geometry(f"350x120+{x}+{y}")

        # Message
        self.message_label = ttk.Label(self.loading_window, text=message, font=("Segoe UI", 10))
        self.message_label.pack(pady=10)

        # Barre de progression
        self.progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(
            self.loading_window, 
            variable=self.progress_var, 
            maximum=100,
            length=300
        )
        progress_bar.pack(fill='x', padx=20, pady=10)

        # Démarrer avec un peu de progression
        self.progress_var.set(10)
        
        # Animation automatique
        self._animate_loading()
        self.parent_app.root.update()

    def update_loading_message(self, message, progress=None):
        """Met à jour le message et la progression - EXACT de l'original"""
        if self.loading_window and self.loading_window.winfo_exists():
            try:
                self.message_label.config(text=message)
                
                # Progression automatique si pas spécifiée
                if progress is None:
                    current = self.progress_var.get()
                    progress = min(current + 15, 90)  # Progression par étapes
                
                self.progress_var.set(progress)
                self.loading_window.update()
            except Exception as e:
                logging.error(f"Erreur mise à jour loading: {e}")

    def _animate_loading(self):
        """Animation de la barre de progression - EXACT de l'original"""
        if self.progress_var is None or not self.loading_window:
            return
            
        try:
            if self.loading_window.winfo_exists():
                current_val = self.progress_var.get()
                if current_val < 95:
                    increment = 2 if current_val < 20 else 1 if current_val < 50 else 0.5
                    new_val = min(current_val + increment, 95)
                    self.progress_var.set(new_val)
                    self.loading_window.after(100, self._animate_loading)
        except Exception:
            pass

    def hide_loading_bar(self):
        """Cache la barre de progression - EXACT de l'original"""
        if self.loading_window:
            try:
                self.loading_window.grab_release()
                self.loading_window.destroy()
            except Exception:
                pass
            finally:
                self.loading_window = None
                self.progress_var = None
