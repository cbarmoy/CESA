"""Unified sleep-scoring backends for YASA and U-Sleep."""

from __future__ import annotations

from dataclasses import dataclass
import os
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence

import mne
import numpy as np
import pandas as pd

StageMethod = Literal["yasa", "usleep", "pftsleep"]
ProgressCallback = Optional[Callable[[str, dict[str, Any]], None]]


@dataclass
class SleepScorer:
    """Sleep scoring facade with interchangeable backends."""

    method: StageMethod = "yasa"
    eeg_candidates: Sequence[str] = ()
    eog_candidates: Sequence[str] = ()
    emg_candidates: Sequence[str] = ()
    epoch_length: float = 30.0
    target_sfreq: float = 100.0
    progress_cb: ProgressCallback = None
    yasa_age: Optional[int] = None
    yasa_male: Optional[bool] = None
    yasa_confidence_threshold: float = 0.80

    # U-Sleep specific runtime options
    usleep_checkpoint_path: Optional[str] = None
    usleep_sfreq: float = 128.0
    usleep_device: Optional[str] = None
    usleep_use_eog: bool = True
    usleep_zscore: bool = True

    # PFTSleep-specific options
    pft_models_dir: Optional[str] = None
    pft_device: str = "auto"
    pft_hf_token: Optional[str] = None
    pft_eeg_channel: Optional[str] = None
    pft_eog_channel: Optional[str] = None
    pft_emg_channel: Optional[str] = None
    pft_ecg_channel: Optional[str] = None

    _CLASS_TO_STAGE = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}

    def score(self, raw: mne.io.BaseRaw) -> pd.DataFrame:
        """Run selected backend and return a standard scoring DataFrame."""
        if not isinstance(raw, mne.io.BaseRaw):
            raise TypeError("SleepScorer.score expects an MNE Raw object.")
        if self.epoch_length <= 0:
            raise ValueError("epoch_length must be > 0.")

        method = str(self.method).lower().strip()
        if method == "yasa":
            return self._score_yasa(raw)
        if method == "usleep":
            return self._score_usleep(raw)
        if method == "pftsleep":
            return self._score_pftsleep(raw)
        raise ValueError(f"Unsupported sleep scoring method: {self.method}")

    def _score_yasa(self, raw: mne.io.BaseRaw) -> pd.DataFrame:
        """Run YASA sleep staging directly via the official API."""
        self._emit("start", {"backend": "yasa"})
        try:
            import yasa  # pyright: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Le package 'yasa' n'est pas installé. "
                "Installez-le pour utiliser le backend YASA."
            ) from exc

        eeg_name = self._pick_first_available_channel(
            raw,
            self.eeg_candidates,
            kind="EEG",
            required=True,
        )
        eog_name = self._pick_first_available_channel(
            raw,
            self.eog_candidates,
            kind="EOG",
            required=False,
        )
        emg_name = self._pick_first_available_channel(
            raw,
            self.emg_candidates,
            kind="EMG",
            required=False,
        )
        metadata = self._build_yasa_metadata()

        self._emit("eeg_selected", {"eeg": eeg_name})
        if eog_name:
            self._emit("eog_selected", {"eog": [eog_name]})
        self._emit(
            "staging_initialized",
            {
                "mode": "yasa",
                "eeg": eeg_name,
                "eog": eog_name,
                "emg": emg_name,
                "metadata": bool(metadata),
                "internal_target_sfreq": float(self.target_sfreq),
            },
        )

        selected_channels = [eeg_name]
        if eog_name and eog_name not in selected_channels:
            selected_channels.append(eog_name)
        if emg_name and emg_name not in selected_channels:
            selected_channels.append(emg_name)

        self._emit("filters", {"selected_channels": list(selected_channels)})
        try:
            # Important: pick before load_data to avoid preloading the full EDF in memory.
            raw_work = raw.copy().pick(selected_channels).load_data()
        except MemoryError as exc:
            raise RuntimeError(
                "Mémoire insuffisante pour charger les canaux nécessaires au scoring YASA. "
                "Le backend YASA ne charge pourtant que les canaux sélectionnés (EEG/EOG/EMG). "
                "Fermez d'autres applications ou relancez CESA puis réessayez."
            ) from exc
        except Exception as exc:
            err_msg = str(exc).lower()
            if "unable to allocate" in err_msg or "memory" in err_msg:
                raise RuntimeError(
                    "Mémoire insuffisante pendant la préparation YASA. "
                    "CESA limite désormais le chargement aux seuls canaux utiles, "
                    "mais ce fichier reste trop coûteux pour la mémoire actuellement disponible."
                ) from exc
            raise
        kwargs: dict[str, Any] = {"eeg_name": eeg_name}
        if eog_name:
            kwargs["eog_name"] = eog_name
        if emg_name:
            kwargs["emg_name"] = emg_name
        if metadata:
            kwargs["metadata"] = metadata

        self._emit("predict_begin", {"backend": "yasa"})
        sls = yasa.SleepStaging(raw_work, **kwargs)
        pred = sls.predict()
        self._emit("predict_end", {"backend": "yasa"})

        hypno = getattr(pred, "hypno", pred)
        proba = getattr(pred, "proba", None)
        if proba is None and hasattr(sls, "predict_proba"):
            try:
                proba = sls.predict_proba()
            except Exception:
                proba = None

        stages = self._normalize_stage_labels(hypno)
        times = np.arange(len(stages), dtype=float) * float(self.epoch_length)
        df = pd.DataFrame({"time": times, "stage": stages})

        proba_df = self._extract_yasa_probabilities(proba)
        if proba_df is not None and len(proba_df) == len(df):
            df = pd.concat([df, proba_df.reset_index(drop=True)], axis=1)
            df["confidence"] = proba_df.max(axis=1).astype(float)

        self._emit(
            "done",
            {
                "epochs": int(len(df)),
                "confidence_mean": float(df["confidence"].mean()) if "confidence" in df.columns and len(df) else None,
            },
        )
        return self._validate_output_df(df)

    def _score_usleep(self, raw: mne.io.BaseRaw) -> pd.DataFrame:
        """Run U-Sleep inference via official Python implementation."""
        self._emit("start", {"backend": "usleep"})
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch n'est pas installé. Installez 'torch' pour utiliser U-Sleep."
            ) from exc
        try:
            from braindecode.models import USleep  # pyright: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Le package 'braindecode' est requis pour U-Sleep."
            ) from exc

        if self.usleep_checkpoint_path is None:
            raise RuntimeError(
                "Mode U-Sleep activé mais aucun checkpoint pré-entraîné fourni. "
                "Renseignez usleep_checkpoint_path."
            )

        ckpt_path = Path(self.usleep_checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint U-Sleep introuvable: {ckpt_path}")

        selected_channels = self._select_input_channels(raw)
        self._emit("eeg_selected", {"eeg": selected_channels[0] if selected_channels else None})
        if len(selected_channels) > 1:
            self._emit("eog_selected", {"eog": selected_channels[1:]})
        raw_work = raw.copy().pick(selected_channels).load_data()

        current_sfreq = float(raw_work.info["sfreq"])
        self._emit("resample_begin", {"from_sfreq": current_sfreq, "target_sfreq": self.usleep_sfreq})
        if not np.isclose(current_sfreq, self.usleep_sfreq, rtol=0.0, atol=1e-3):
            raw_work.resample(float(self.usleep_sfreq), npad="auto")
        self._emit("resample_end", {"target_sfreq": float(raw_work.info["sfreq"])})

        x = raw_work.get_data().astype(np.float32, copy=False)
        if x.ndim != 2:
            raise RuntimeError(f"Unexpected input shape from Raw.get_data(): {x.shape}")
        if bool(self.usleep_zscore):
            x = self._zscore_channels(x)
        n_channels, n_samples = x.shape
        if n_samples < int(self.epoch_length * float(raw_work.info["sfreq"])):
            raise RuntimeError(
                f"Signal trop court pour une époque ({n_samples} samples)."
            )

        device_name = self.usleep_device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_name)
        model = self._build_usleep_model(USleep, n_channels=n_channels)
        state = torch.load(str(ckpt_path), map_location=device)
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        self._emit("staging_initialized", {"mode": "usleep", "device": str(device)})

        x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)
        self._emit("predict_begin", {"shape": tuple(x_tensor.shape)})
        with torch.no_grad():
            logits = model(x_tensor)
        self._emit("predict_end", {"shape": tuple(logits.shape) if hasattr(logits, "shape") else "unknown"})

        y_pred = self._decode_predictions(logits, n_samples=n_samples, sfreq=float(raw_work.info["sfreq"]))
        stages = [self._CLASS_TO_STAGE.get(int(k), "U") for k in y_pred]
        times = np.arange(len(stages), dtype=float) * float(self.epoch_length)
        df = pd.DataFrame({"time": times, "stage": stages})
        self._emit("done", {"epochs": int(len(df))})
        return self._validate_output_df(df)

    def _score_pftsleep(self, raw: mne.io.BaseRaw) -> pd.DataFrame:
        """Run PFTSleep foundational transformer on the current EDF file."""
        self._emit("start", {"backend": "pftsleep"})
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch n'est pas installé. Installez 'torch' pour utiliser PFTSleep."
            ) from exc
        try:
            from pftsleep.inference import (  # type: ignore
                download_pftsleep_models,
                infer_on_edf,
                HYPNOGRAM_EPOCH_SECONDS_DEFAULT,
                view_edf_channels,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Le package 'PFTSleep' n'est pas installé. Installez-le avec 'pip install PFTSleep'."
            ) from exc

        # Résoudre le chemin EDF depuis le Raw quand c'est possible.
        edf_path = None
        try:
            # MNE stocke souvent le chemin dans _filenames ou filenames
            filenames = getattr(raw, "filenames", None) or getattr(raw, "_filenames", None)
            if filenames:
                edf_path = filenames[0]
        except Exception:
            edf_path = None
        if not edf_path:
            raise RuntimeError(
                "Le backend PFTSleep nécessite le chemin EDF d'origine, "
                "qui n'a pas pu être résolu depuis l'objet Raw."
            )
        edf_path = str(edf_path)

        # Résoudre le dossier modèles + téléchargement si nécessaire.
        # Par défaut, on pointe vers le dossier PFTSleep/ à la racine du projet,
        # où tu as déjà placé pft_sleep_encoder.ckpt et pft_sleep_classifier.ckpt.
        project_root = Path(__file__).resolve().parent.parent
        default_models_dir = project_root / "PFTSleep"
        models_dir = self.pft_models_dir or str(default_models_dir)
        models_dir_path = Path(models_dir)
        models_dir_path.mkdir(parents=True, exist_ok=True)
        encoder_ckpt = models_dir_path / "pft_sleep_encoder.ckpt"
        classifier_ckpt = models_dir_path / "pft_sleep_classifier.ckpt"
        if not encoder_ckpt.exists() or not classifier_ckpt.exists():
            logging.info("[PFTSLEEP] Téléchargement des modèles dans %s", models_dir_path)
            # Priorité: champ UI pft_hf_token, puis variables d'environnement.
            token = (self.pft_hf_token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or "").strip() or None
            try:
                download_pftsleep_models(write_dir=str(models_dir_path), token=token)
            except Exception as e:
                err_msg = str(e).lower()
                if "401" in err_msg or "gated" in err_msg or "authenticated" in err_msg or ("access" in err_msg and "restricted" in err_msg):
                    raise RuntimeError(
                        "Les modèles PFTSleep sont sur un dépôt Hugging Face restreint.\n\n"
                        "1. Demandez l'accès : https://huggingface.co/benmfox/PFTSleep\n"
                        "2. Créez un token (Read) : https://huggingface.co/settings/tokens\n"
                        "3. Soit définissez la variable d'environnement HUGGINGFACE_TOKEN, "
                        "soit, dans CESA : menu Scoring de sommeil → Paramètres scoring… → section PFTSleep → "
                        "collez le token dans « Token HF (optionnel) », puis relancez le scoring."
                    ) from e
                raise

        # Choix device
        if self.pft_device and self.pft_device.lower() not in {"auto"}:
            device = self.pft_device
        else:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Mapping des canaux: auto-détection avec possibilité de forcer depuis les champs pft_*_channel.
        try:
            available = [c.upper() for c in view_edf_channels(edf_path, uppercase=True)]
        except Exception:
            available = []

        def _pick(preferred: Sequence[str]) -> str:
            for name in preferred:
                if name.upper() in available:
                    return name.upper()
            return "dummy"

        eeg = (self.pft_eeg_channel or "").strip().upper() or _pick(
            ["C4-M1", "C3-M2", "FPZ-CZ", "PZ-OZ"]
        )
        left_eog = (self.pft_eog_channel or "").strip().upper() or _pick(
            ["E1-M2", "E2-M1", "EOG L", "EOG LEFT", "EOG GAUCHE"]
        )
        chin_emg = (self.pft_emg_channel or "").strip().upper() or _pick(
            ["CHIN1-CHIN2", "CHIN1-CHIN3", "EMG CHIN", "CHIN", "MENTON"]
        )
        ecg = (self.pft_ecg_channel or "").strip().upper() or _pick(
            ["ECG", "ECG1", "ECG2", "EKG"]
        )
        # Canaux supplémentaires optionnels: utiliser 'dummy' si absent.
        spo2 = _pick(["SPO2", "SAO2"])
        abdomen_rr = _pick(["ABDO", "ABD", "ABDOMEN"])
        thoracic_rr = _pick(["THOR", "THX", "THORAX"])

        try:
            logging.info(
                "[PFTSLEEP] Inference on EDF=%s, device=%s, eeg=%s, eog=%s, emg=%s, ecg=%s",
                edf_path,
                device,
                eeg,
                left_eog,
                chin_emg,
                ecg,
            )
            logits = infer_on_edf(
                edf_file_path=edf_path,
                eeg_channel=eeg or "dummy",
                left_eog_channel=left_eog or "dummy",
                chin_emg_channel=chin_emg or "dummy",
                ecg_channel=ecg or "dummy",
                spo2_channel=spo2,
                abdomen_rr_channel=abdomen_rr,
                thoracic_rr_channel=thoracic_rr,
                models_dir=str(models_dir_path),
                device=device,
            )
        except Exception as exc:
            raise RuntimeError(f"[PFTSLEEP] Echec de l'inférence: {exc}") from exc

        # logits: [5, n_epochs] où les 5 classes sont [Wake,N1,N2,N3,REM].
        arr = self._to_numpy(logits)
        if arr.ndim != 2 or arr.shape[0] != 5:
            raise RuntimeError(f"[PFTSLEEP] Sortie inattendue: shape={arr.shape}")

        # PFTSleep donne déjà une prédiction par époque (30 s).
        epoch_sec = float(
            getattr(HYPNOGRAM_EPOCH_SECONDS_DEFAULT, "value", HYPNOGRAM_EPOCH_SECONDS_DEFAULT)
        ) if hasattr(HYPNOGRAM_EPOCH_SECONDS_DEFAULT, "value") else float(HYPNOGRAM_EPOCH_SECONDS_DEFAULT)
        sec_per_epoch = max(epoch_sec, 1.0)

        # Argmax directement sur l'axe des classes pour chaque époque.
        epoch_labels = arr.argmax(axis=0).astype(int).tolist()

        # Tronquer selon la durée réelle de l'enregistrement.
        try:
            dur_sec = float(len(raw.times) / raw.info["sfreq"])
            max_epochs = int(dur_sec // sec_per_epoch)
            if max_epochs > 0:
                epoch_labels = epoch_labels[:max_epochs]
        except Exception:
            pass

        stages = [self._CLASS_TO_STAGE.get(int(k), "U") for k in epoch_labels]
        times = np.arange(len(stages), dtype=float) * sec_per_epoch
        df = pd.DataFrame({"time": times, "stage": stages})
        self._emit("done", {"epochs": int(len(df))})
        return self._validate_output_df(df)

    def _select_input_channels(self, raw: mne.io.BaseRaw) -> list[str]:
        """Pick channels for U-Sleep while matching current project conventions."""
        available = list(raw.ch_names)
        chosen: list[str] = []
        for name in self.eeg_candidates:
            if name in available:
                chosen.append(name)
                break
        if not chosen:
            raise RuntimeError("Aucun canal EEG compatible trouvé pour U-Sleep.")

        if self.usleep_use_eog:
            for name in self.eog_candidates:
                if name in available and name not in chosen:
                    chosen.append(name)
                    break

        return chosen

    @staticmethod
    def _pick_first_available_channel(
        raw: mne.io.BaseRaw,
        candidates: Sequence[str],
        *,
        kind: str,
        required: bool,
    ) -> Optional[str]:
        available = list(raw.ch_names)
        normalized = {name.upper(): name for name in available}
        for name in candidates:
            candidate = str(name).strip()
            if not candidate:
                continue
            if candidate in available:
                return candidate
            resolved = normalized.get(candidate.upper())
            if resolved is not None:
                return resolved
        if required:
            raise RuntimeError(
                f"Aucun canal {kind} compatible trouvé pour YASA. "
                f"Candidats testés: {', '.join(map(str, candidates)) or 'aucun'}."
            )
        return None

    def _build_yasa_metadata(self) -> Optional[dict[str, Any]]:
        metadata: dict[str, Any] = {}
        if self.yasa_age is not None:
            try:
                metadata["age"] = int(self.yasa_age)
            except Exception:
                pass
        if self.yasa_male is not None:
            metadata["male"] = bool(self.yasa_male)
        return metadata or None

    @staticmethod
    def _normalize_stage_labels(hypno: Any) -> list[str]:
        if isinstance(hypno, pd.DataFrame):
            if "stage" in hypno.columns:
                values = hypno["stage"].tolist()
            elif hypno.shape[1] >= 1:
                values = hypno.iloc[:, 0].tolist()
            else:
                values = []
        elif isinstance(hypno, pd.Series):
            values = hypno.tolist()
        else:
            values = list(np.asarray(hypno).reshape(-1).tolist())

        mapping = {
            "WAKE": "W",
            "W": "W",
            "N1": "N1",
            "N2": "N2",
            "N3": "N3",
            "REM": "R",
            "R": "R",
            "U": "U",
            "UNKNOWN": "U",
        }
        out: list[str] = []
        for value in values:
            stage = str(value).upper().strip()
            out.append(mapping.get(stage, stage if stage in {"W", "N1", "N2", "N3", "R"} else "U"))
        return out

    def _extract_yasa_probabilities(self, proba: Any) -> Optional[pd.DataFrame]:
        if proba is None:
            return None
        if isinstance(proba, pd.DataFrame):
            df = proba.copy()
        elif isinstance(proba, pd.Series):
            df = proba.to_frame()
        else:
            arr = self._to_numpy(proba)
            if arr.ndim != 2:
                return None
            if arr.shape[0] == 5 and arr.shape[1] != 5:
                arr = arr.T
            df = pd.DataFrame(arr)

        canonical = ["W", "N1", "N2", "N3", "R"]
        name_map = {
            "WAKE": "W",
            "W": "W",
            "N1": "N1",
            "N2": "N2",
            "N3": "N3",
            "REM": "R",
            "R": "R",
        }
        renamed: dict[Any, str] = {}
        for col in df.columns:
            mapped = name_map.get(str(col).upper().strip())
            if mapped:
                renamed[col] = mapped
        if renamed:
            df = df.rename(columns=renamed)
        elif df.shape[1] == len(canonical):
            df.columns = canonical
        else:
            return None

        out = pd.DataFrame(index=df.index)
        for stage in canonical:
            if stage in df.columns:
                out[f"proba_{stage}"] = pd.to_numeric(df[stage], errors="coerce")
        if out.empty:
            return None
        return out

    def _build_usleep_model(self, cls: type, n_channels: int):
        """Instantiate braindecode USleep defensively across API versions."""
        sig = inspect.signature(cls)
        candidate_kwargs = (
            {"n_chans": n_channels, "n_classes": 5, "sfreq": self.usleep_sfreq},
            {"in_chans": n_channels, "n_classes": 5, "sfreq": self.usleep_sfreq},
            {"n_channels": n_channels, "n_classes": 5, "sfreq": self.usleep_sfreq},
            {"n_chans": n_channels, "n_outputs": 5, "sfreq": self.usleep_sfreq},
        )
        for kwargs in candidate_kwargs:
            filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
            if not filtered:
                continue
            try:
                return cls(**filtered)
            except Exception:
                continue
        raise RuntimeError(
            "Impossible d'instancier USleep (API braindecode inattendue). "
            "Vérifiez la version installée."
        )

    def _decode_predictions(self, logits: Any, *, n_samples: int, sfreq: float) -> np.ndarray:
        """Convert model logits/probabilities into one class per epoch."""
        arr = self._to_numpy(logits)
        if arr.ndim == 3:
            if arr.shape[1] == 5:
                pred = arr.argmax(axis=1).reshape(-1)
            elif arr.shape[2] == 5:
                pred = arr.argmax(axis=2).reshape(-1)
            else:
                raise RuntimeError(f"Sortie U-Sleep 3D inattendue: {arr.shape}")
        elif arr.ndim == 2:
            if arr.shape[1] == 5:
                pred = arr.argmax(axis=1)
            elif arr.shape[0] == 5:
                pred = arr.argmax(axis=0)
            else:
                raise RuntimeError(f"Sortie U-Sleep 2D inattendue: {arr.shape}")
        elif arr.ndim == 1:
            pred = arr.astype(np.int64)
        else:
            raise RuntimeError(f"Sortie U-Sleep non supportée: {arr.shape}")

        pred = np.asarray(pred, dtype=np.int64).reshape(-1)
        epoch_samples = int(round(float(self.epoch_length) * float(sfreq)))
        n_epochs = int(n_samples // epoch_samples)
        if n_epochs <= 0:
            raise RuntimeError("Nombre d'époques invalide après découpage.")

        if len(pred) == n_epochs:
            return pred
        if len(pred) == n_samples:
            return self._downsample_samplewise_predictions(pred, epoch_samples, n_epochs)
        if len(pred) > n_epochs:
            idx = np.linspace(0, len(pred) - 1, n_epochs, dtype=int)
            return pred[idx]

        raise RuntimeError(
            f"Sortie U-Sleep trop courte ({len(pred)} valeurs) pour {n_epochs} époques."
        )

    @staticmethod
    def _downsample_samplewise_predictions(pred: np.ndarray, epoch_samples: int, n_epochs: int) -> np.ndarray:
        out = np.zeros(n_epochs, dtype=np.int64)
        trimmed = pred[: n_epochs * epoch_samples]
        for i in range(n_epochs):
            chunk = trimmed[i * epoch_samples : (i + 1) * epoch_samples]
            values, counts = np.unique(chunk, return_counts=True)
            out[i] = int(values[int(np.argmax(counts))])
        return out

    @staticmethod
    def _zscore_channels(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        means = x.mean(axis=1, keepdims=True)
        stds = x.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-6, 1.0, stds)
        return (x - means) / stds

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    @staticmethod
    def _validate_output_df(df: Any) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise RuntimeError(f"Le backend de scoring doit renvoyer un DataFrame, reçu: {type(df)!r}")
        if "time" not in df.columns or "stage" not in df.columns:
            raise RuntimeError("Le scoring doit contenir les colonnes 'time' et 'stage'.")
        extra_cols = [col for col in df.columns if col not in {"time", "stage"}]
        out = df[["time", "stage"] + extra_cols].copy()
        out["time"] = pd.to_numeric(out["time"], errors="coerce")
        out["stage"] = out["stage"].astype(str).str.upper().str.strip()
        for col in extra_cols:
            if str(col).lower().startswith("proba_") or str(col).lower() == "confidence":
                out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["time"]).reset_index(drop=True)
        return out

    def _emit(self, stage: str, payload: dict[str, Any]) -> None:
        if self.progress_cb is None:
            return
        try:
            self.progress_cb(stage, payload)
        except Exception as exc:
            logging.debug("SleepScorer progress callback failed: %s", exc)
