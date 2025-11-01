"""Utilities to build multiscale min/max pyramids for PSG data."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import zarr
from numcodecs import Blosc

try:  # pragma: no cover - optional dependency
    from mne.io import BaseRaw  # type: ignore
    import mne
except Exception:  # pragma: no cover - optional dependency
    BaseRaw = object  # type: ignore
    mne = None  # type: ignore

from .store import LevelDescriptor, MultiscaleStore, MultiscaleMetadata, open_multiscale


_DEFAULT_COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


@dataclass
class _LevelState:
    descriptor: LevelDescriptor
    residual_ds: zarr.Array
    residual_valid: int
    residual_buffer: np.ndarray
    cursor: int

    @property
    def bin_size(self) -> int:
        return self.descriptor.bin_size

    @property
    def dataset(self) -> zarr.Array:
        return self.descriptor.dataset

    @property
    def n_channels(self) -> int:
        return self.descriptor.dataset.shape[0]

    def processed_samples(self) -> int:
        return self.cursor * self.bin_size + self.residual_valid

    def append(self, samples: np.ndarray) -> None:
        if samples.shape[0] != self.n_channels:
            raise ValueError("Channel mismatch in level writer append")

        if self.residual_valid:
            data = np.concatenate(
                [self.residual_buffer[:, : self.residual_valid], samples], axis=1
            )
        else:
            data = samples

        data_length = data.shape[1]

        if data_length < self.bin_size:
            self.residual_buffer.fill(0.0)
            if data_length:
                self.residual_buffer[:, :data_length] = data
            self.residual_valid = data_length
            return

        n_bins = data_length // self.bin_size
        usable = data[:, : n_bins * self.bin_size]
        remainder = data[:, n_bins * self.bin_size :]

        reshaped = usable.reshape(self.n_channels, n_bins, self.bin_size)
        mins = reshaped.min(axis=2)
        maxs = reshaped.max(axis=2)

        block = np.empty((self.n_channels, n_bins, 2), dtype=np.float32)
        block[:, :, 0] = mins
        block[:, :, 1] = maxs

        start = self.cursor
        stop = start + n_bins
        self.dataset.oindex[:, start:stop, :] = block
        self.cursor = stop

        remainder_len = remainder.shape[1]
        self.residual_buffer.fill(0.0)
        if remainder_len:
            self.residual_buffer[:, :remainder_len] = remainder
        self.residual_valid = remainder_len

    def flush(self) -> None:
        if not self.residual_valid:
            return
        padded = self.residual_buffer[:, : self.residual_valid]
        pad_width = self.bin_size - self.residual_valid
        if pad_width > 0:
            last = padded[:, -1][:, None]
            padded = np.concatenate([padded, np.repeat(last, pad_width, axis=1)], axis=1)

        reshaped = padded.reshape(self.n_channels, 1, self.bin_size)
        mins = reshaped.min(axis=2)
        maxs = reshaped.max(axis=2)
        block = np.empty((self.n_channels, 1, 2), dtype=np.float32)
        block[:, :, 0] = mins
        block[:, :, 1] = maxs
        self.dataset.oindex[:, self.cursor : self.cursor + 1, :] = block
        self.cursor += 1
        self.residual_buffer.fill(0.0)
        self.residual_valid = 0

    def persist_state(self) -> None:
        padded = np.zeros((self.n_channels, self.bin_size), dtype=np.float32)
        if self.residual_valid:
            padded[:, : self.residual_valid] = self.residual_buffer[:, : self.residual_valid]

        self.residual_ds[...] = padded
        self.residual_ds.attrs["valid"] = self.residual_valid
        self.dataset.attrs["cursor"] = self.cursor

    def mark_complete(self) -> None:
        self.dataset.attrs["completed"] = True
        self.residual_ds.attrs["valid"] = 0


def build_pyramid(
    raw_source: BaseRaw | str | Path,
    out_ms_path: str | Path,
    *,
    chunk_seconds: int = 20,
    levels: Sequence[int] | None = None,
    dtype: np.dtype = np.float32,
    resume: bool = True,
    selected_channels: Sequence[str] | None = None,
) -> MultiscaleStore:
    """Build (or resume) a multiscale min/max pyramid from raw PSG data.
    
    Args:
        raw_source: MNE Raw object or path to EDF file
        out_ms_path: Path to output Zarr multiscale store
        chunk_seconds: Size of temporal chunks in seconds
        levels: Custom bin sizes for pyramid levels (optional)
        dtype: Data type for storage (default: float32)
        resume: Resume from previous incomplete build (default: True)
        selected_channels: List of channel names to include (optional, all channels if None)
    
    Returns:
        MultiscaleStore: The constructed multiscale store
    """

    raw = _resolve_raw(raw_source)
    fs = float(raw.info["sfreq"])
    
    # Filtrer les canaux si demandé
    if selected_channels is not None:
        # Vérifier que tous les canaux demandés existent
        available_channels = set(raw.ch_names)
        missing_channels = set(selected_channels) - available_channels
        if missing_channels:
            print(f"⚠️  Canaux manquants ignorés: {missing_channels}", flush=True)
        
        # Ne garder que les canaux disponibles
        channel_names = [ch for ch in selected_channels if ch in available_channels]
        
        if not channel_names:
            raise ValueError("Aucun canal valide fourni dans selected_channels")
        
        # Créer un subset du Raw avec uniquement les canaux sélectionnés
        raw = raw.copy().pick_channels(channel_names)
        print(f"✅ Sélection de {len(channel_names)} canaux sur {len(available_channels)} disponibles", flush=True)
    else:
        channel_names = list(raw.ch_names)
    
    n_channels = len(channel_names)
    n_samples = raw.n_times

    dtype = np.dtype(dtype)
    if dtype != np.float32:
        raise ValueError("Only float32 dtype is currently supported for pyramid build")
    ms_path = Path(out_ms_path)
    ms_path.parent.mkdir(parents=True, exist_ok=True)

    if ms_path.exists() and not resume:
        shutil.rmtree(ms_path)

    if levels is None:
        levels = _default_levels(n_samples)
    else:
        levels = sorted({int(b) for b in levels})
        if not levels or levels[0] != 1:
            levels = [1] + [lvl for lvl in levels if lvl != 1]

    if resume and ms_path.exists():
        cleanup_reasons: list[str] = []
        try:
            existing_store = open_multiscale(ms_path)
            metadata = existing_store.metadata

            if len(metadata.level_bin_sizes) != len(levels):
                cleanup_reasons.append(
                    f"niveaux existants={len(metadata.level_bin_sizes)} (attendu={len(levels)})"
                )
            if metadata.total_samples != n_samples:
                cleanup_reasons.append(
                    f"échantillons existants={metadata.total_samples:,} (attendu={n_samples:,})"
                )
            if not math.isclose(metadata.sampling_frequency, fs, rel_tol=1e-6):
                cleanup_reasons.append(
                    f"fs existant={metadata.sampling_frequency} (attendu={fs})"
                )
            if list(metadata.channel_names) != channel_names:
                cleanup_reasons.append(
                    "liste de canaux différente"
                )

            if cleanup_reasons:
                reason_msg = "; ".join(cleanup_reasons)
                print(
                    f"⚠️  Fichier de navigation incompatible détecté ({reason_msg})",
                    flush=True,
                )
                print(
                    f"🗑️  Suppression automatique de {ms_path}",
                    flush=True,
                )
                try:
                    store_backend = getattr(existing_store.group, "store", None)
                    if store_backend is not None and hasattr(store_backend, "close"):
                        store_backend.close()
                finally:
                    shutil.rmtree(ms_path)
                print(
                    "✅ Ancien fichier supprimé, reconstruction en cours...",
                    flush=True,
                )
        except Exception as inspect_error:
            print(
                f"⚠️  Impossible de vérifier le fichier existant ({inspect_error}), reconstruction forcée",
                flush=True,
            )
            shutil.rmtree(ms_path)
            print(
                "✅ Ancien fichier supprimé, reconstruction en cours...",
                flush=True,
            )

    requested_chunk_seconds = chunk_seconds

    # Ajuster automatiquement la taille des chunks pour limiter l'utilisation mémoire
    # Cible : ~200-500 MB par chunk (éviter les pics > 1 GB)
    TARGET_CHUNK_MB = 300  # MB par chunk
    bytes_per_sample = 4  # float32
    bytes_per_channel_per_second = fs * bytes_per_sample
    max_seconds_per_chunk = (
        TARGET_CHUNK_MB * 1024 * 1024 / (n_channels * bytes_per_channel_per_second)
    ) if n_channels else requested_chunk_seconds

    # Limiter entre 1 et chunk_seconds (ne jamais dépasser la valeur demandée)
    chunk_seconds = max(1, min(int(max_seconds_per_chunk), requested_chunk_seconds))
    chunk_samples = max(1, int(round(chunk_seconds * fs)))

    # Estimation mémoire
    estimated_mb_per_chunk = (
        (n_channels * chunk_samples * bytes_per_sample) / (1024 * 1024)
        if n_channels
        else 0.0
    )

    _prepare_store(
        ms_path,
        fs=fs,
        channel_names=channel_names,
        n_samples=n_samples,
        dtype=dtype,
        levels=levels,
        chunk_seconds=chunk_seconds,
    )

    # Ouvrir en mode écriture pour la construction, pas en lecture seule
    root_group = zarr.open_group(str(ms_path), mode="a")
    store = _open_multiscale_writable(ms_path, root_group)
    level_states = _initialise_level_states(store, n_channels)

    processed_counts = {state.processed_samples() for state in level_states}
    if len(processed_counts) != 1:
        raise RuntimeError("Inconsistent resume state across pyramid levels")
    processed_samples = processed_counts.pop()

    if processed_samples >= n_samples:
        for state in level_states:
            state.flush()
            state.persist_state()
            state.mark_complete()
        return store

    # Variables pour le suivi de progression
    import time
    start_time = time.time()
    total_chunks = math.ceil((n_samples - processed_samples) / chunk_samples)
    chunk_counter = 0
    last_checkpoint_percent = -10  # Pour afficher tous les 10%
    
    duration_seconds = n_samples / fs
    duration_minutes = duration_seconds / 60
    
    print(f"🚀 Démarrage de la construction de la pyramide", flush=True)
    print(f"   📊 Données : {n_channels} canaux, {n_samples:,} échantillons ({duration_minutes:.1f} min @ {fs} Hz)", flush=True)
    print(f"   🔧 Niveaux : {len(levels)} niveaux (bin_sizes: {levels})", flush=True)
    print(
        f"   📦 Chunks : {total_chunks} chunks de {chunk_seconds}s (~{estimated_mb_per_chunk:.1f} MB/chunk)",
        flush=True,
    )
    if chunk_seconds < requested_chunk_seconds:
        print(
            f"   ⚠️  Taille des chunks ajustée : {requested_chunk_seconds}s → {chunk_seconds}s (optimisation mémoire)",
            flush=True,
        )
    if processed_samples > 0:
        print(f"   ♻️  Reprise depuis : {processed_samples:,} échantillons ({processed_samples/fs/60:.1f} min)", flush=True)
    print(flush=True)

    while processed_samples < n_samples:
        stop = min(processed_samples + chunk_samples, n_samples)
        
        # Checkpoint : progression en %
        current_percent = int((processed_samples / n_samples) * 100)
        if current_percent >= last_checkpoint_percent + 10 or chunk_counter == 0:
            elapsed = time.time() - start_time
            progress_ratio = processed_samples / n_samples if processed_samples > 0 else 0.01
            estimated_total = elapsed / progress_ratio if progress_ratio > 0 else 0
            remaining = estimated_total - elapsed
            
            # Calcul de la vitesse
            if elapsed > 0:
                speed_samples_per_sec = processed_samples / elapsed
                speed_minutes_per_sec = (processed_samples / fs) / elapsed
            else:
                speed_samples_per_sec = 0
                speed_minutes_per_sec = 0
            
            print(f"⏳ CHECKPOINT [{current_percent:3d}%] - Chunk {chunk_counter}/{total_chunks}", flush=True)
            print(f"   📍 Position : {processed_samples:,} / {n_samples:,} échantillons ({processed_samples/fs/60:.1f} / {duration_minutes:.1f} min)", flush=True)
            print(f"   ⚡ Vitesse : {speed_samples_per_sec:,.0f} éch/s ({speed_minutes_per_sec:.1f}x temps réel)", flush=True)
            print(f"   ⏱️  Écoulé : {elapsed:.1f}s | Restant : ~{remaining:.1f}s", flush=True)
            print(flush=True)
            
            last_checkpoint_percent = current_percent
        
        # Lecture des données
        data = raw.get_data(start=processed_samples, stop=stop)
        # Force une copie pour éviter les erreurs "read-only" depuis MNE
        data_uv = (data * 1e6).astype(np.float32, copy=True)

        for state in level_states:
            state.append(data_uv)

        processed_samples = stop
        chunk_counter += 1

        for state in level_states:
            state.persist_state()

    # Checkpoint final avant flush
    print(f"✅ CHECKPOINT [100%] - Traitement terminé", flush=True)
    print(f"   📍 Position : {processed_samples:,} / {n_samples:,} échantillons", flush=True)
    print(flush=True)
    print(f"💾 Finalisation de la pyramide...", flush=True)
    
    for state in level_states:
        state.flush()
        state.persist_state()
        state.mark_complete()

    final_group = zarr.open_group(str(ms_path), mode="a")
    final_group.attrs["build_complete"] = True
    final_group.attrs["chunk_seconds"] = chunk_seconds
    final_group.attrs["chunk_seconds_requested"] = requested_chunk_seconds

    total_time = time.time() - start_time
    compression_ratio = _calculate_compression_ratio(ms_path)
    
    print(f"✅ Pyramide construite avec succès !", flush=True)
    print(f"   ⏱️  Durée totale : {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    print(f"   💾 Compression : ~{compression_ratio:.1f}:1", flush=True)
    print(f"   📁 Emplacement : {ms_path}", flush=True)
    print(flush=True)

    store = open_multiscale(ms_path)
    return store


def _resolve_raw(raw_source: BaseRaw | str | Path) -> BaseRaw:
    if isinstance(raw_source, BaseRaw):
        return raw_source

    path = Path(raw_source)
    if not path.exists():
        raise FileNotFoundError(path)

    if mne is None:
        raise RuntimeError("mne is required to load raw PSG files from paths")

    reader = _select_reader(path)
    raw = reader(str(path), preload=False)
    return raw


def _select_reader(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".fif":
        return mne.io.read_raw_fif
    if suffix in {".edf", ".bdf"}:
        return mne.io.read_raw_edf
    raise ValueError(f"Unsupported raw format for pyramid build: {path.suffix}")


def _default_levels(n_samples: int) -> List[int]:
    """
    Génère les niveaux de pyramide nécessaires.
    Limite à ~13-14 niveaux pour équilibrer performance et stockage.
    """
    if n_samples <= 0:
        return [1]
    
    levels = [1]
    
    # Générer des niveaux jusqu'à couvrir environ 10-30 secondes en un seul bin
    # Pour un signal PSG typique (256-512 Hz, plusieurs heures), cela donne ~13 niveaux
    # Au-delà, les niveaux deviennent inutiles pour la navigation
    MAX_USEFUL_LEVELS = 14  # [1, 2, 4, ..., 8192] = 14 niveaux
    
    while len(levels) < MAX_USEFUL_LEVELS:
        next_level = levels[-1] * 2
        # S'arrêter si le prochain niveau dépasse la moitié du signal
        # (au-delà, on n'a plus assez de résolution pour zoomer efficacement)
        if next_level >= n_samples // 2:
            break
        levels.append(next_level)
    
    return levels


def _open_multiscale_writable(path: Path, group: zarr.Group) -> MultiscaleStore:
    """Open a multiscale store in writable mode (for building)."""
    attrs = dict(group.attrs)
    
    try:
        fs = float(attrs["fs"])
        channel_names = list(attrs["channel_names"])
        levels_list = [int(v) for v in attrs["levels"]]
        total_samples = int(attrs["n_samples"])
        dtype = np.dtype(attrs.get("dtype", "float32"))
    except KeyError as exc:
        raise RuntimeError(f"Missing attribute in multiscale store: {exc}") from exc

    metadata = MultiscaleMetadata(
        sampling_frequency=fs,
        channel_names=tuple(channel_names),
        level_bin_sizes=tuple(levels_list),
        total_samples=total_samples,
        dtype=dtype,
    )

    level_group = group["levels"]
    descriptors = []
    for bin_size in levels_list:
        level_name = f"lvl{bin_size}"
        dataset = level_group[level_name]
        n_bins = dataset.shape[1]
        chunk_bins = dataset.chunks[1]  # Récupérer chunk_bins depuis les chunks Zarr
        descriptors.append(
            LevelDescriptor(bin_size=bin_size, dataset=dataset, n_bins=n_bins, chunk_bins=chunk_bins)
        )

    return MultiscaleStore(
        root_path=path, group=group, metadata=metadata, levels=descriptors
    )


def _prepare_store(
    path: Path,
    *,
    fs: float,
    channel_names: Sequence[str],
    n_samples: int,
    dtype: np.dtype,
    levels: Sequence[int],
    chunk_seconds: int,
) -> None:
    root = zarr.open_group(str(path), mode="a")

    existing = dict(root.attrs)
    if existing:
        if not _metadata_matches(existing, fs, channel_names, n_samples, dtype, levels):
            raise RuntimeError(
                "Existing multiscale store metadata does not match requested build"
            )
    else:
        root.attrs.update(
            fs=fs,
            channel_names=list(channel_names),
            levels=list(levels),
            n_samples=int(n_samples),
            dtype=str(dtype),
        )

    levels_group = root.require_group("levels")
    residuals_group = root.require_group("residuals")

    n_channels = len(channel_names)

    for bin_size in levels:
        n_bins = math.ceil(n_samples / bin_size)
        chunk_bins = max(1, int(round(chunk_seconds * fs / bin_size)))
        dataset = levels_group.require_dataset(
            name=f"lvl{bin_size}",
            shape=(n_channels, n_bins, 2),
            chunks=(1, chunk_bins, 2),
            dtype=dtype,
            compressor=_DEFAULT_COMPRESSOR,
            overwrite=False,
        )

        if dataset.attrs.get("initialized") is None:
            dataset.attrs["cursor"] = 0
            dataset.attrs["completed"] = False
            dataset.attrs["initialized"] = True

        residual = residuals_group.require_dataset(
            name=f"lvl{bin_size}",
            shape=(n_channels, bin_size),
            chunks=(n_channels, bin_size),
            dtype=dtype,
            compressor=None,
            overwrite=False,
        )
        residual.attrs.setdefault("valid", 0)


def _metadata_matches(
    existing: dict,
    fs: float,
    channel_names: Sequence[str],
    n_samples: int,
    dtype: np.dtype,
    levels: Sequence[int],
) -> bool:
    return (
        float(existing.get("fs", fs)) == fs
        and list(existing.get("channel_names", channel_names)) == list(channel_names)
        and list(existing.get("levels", levels)) == list(levels)
        and int(existing.get("n_samples", n_samples)) == int(n_samples)
        and str(existing.get("dtype", str(dtype))) == str(dtype)
    )


def _initialise_level_states(store: MultiscaleStore, n_channels: int) -> List[_LevelState]:
    root = store.group
    residuals_group = root.require_group("residuals")
    levels: List[_LevelState] = []

    for bin_size in store.available_levels():
        descriptor = store.get_level(bin_size)
        dataset = descriptor.dataset
        cursor = int(dataset.attrs.get("cursor", 0))
        residual_ds = residuals_group[f"lvl{bin_size}"]
        valid = int(residual_ds.attrs.get("valid", 0))
        buffer = np.zeros((n_channels, bin_size), dtype=np.float32)
        if valid:
            buffer[:, :valid] = np.asarray(
                residual_ds.oindex[:, :valid], dtype=np.float32, order="C"
            )
        levels.append(
            _LevelState(
                descriptor=descriptor,
                residual_ds=residual_ds,
                residual_valid=valid,
                residual_buffer=buffer,
                cursor=cursor,
            )
        )

    return levels


def _calculate_compression_ratio(ms_path: Path) -> float:
    """Calculate approximate compression ratio of the multiscale store."""
    try:
        import os
        
        total_size = 0
        for root, dirs, files in os.walk(ms_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
        
        # Lire les métadonnées pour calculer la taille non compressée
        group = zarr.open_group(str(ms_path), mode="r")
        attrs = dict(group.attrs)
        
        n_samples = int(attrs.get("n_samples", 0))
        n_channels = len(attrs.get("channel_names", []))
        
        # Taille brute (float32 = 4 bytes)
        raw_size = n_samples * n_channels * 4
        
        if total_size > 0:
            return raw_size / total_size
        else:
            return 1.0
    except Exception:
        return 1.0


