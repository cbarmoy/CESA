"""Command-line interface for multiscale PSG tools."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

from core.multiscale import build_pyramid
from CESA.eeg_studio_fixed import main as view_application


def _parse_levels(arg: str | None) -> Sequence[int] | None:
    if not arg:
        return None
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    try:
        return [int(p) for p in parts]
    except ValueError as exc:  # pragma: no cover - argument parsing
        raise argparse.ArgumentTypeError(f"Invalid level list: {arg}") from exc


def _cmd_build_pyramid(args: argparse.Namespace) -> int:
    levels = _parse_levels(args.levels)
    build_pyramid(
        raw_source=args.raw,
        out_ms_path=args.out,
        chunk_seconds=args.chunk_seconds,
        levels=levels,
        resume=not args.fresh,
    )
    print(f"✅ Pyramid built at {Path(args.out).resolve()}")
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    return view_application(ms_path=args.ms_path)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PSG multiscale utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build-pyramid",
        help="Pré-calculer une pyramide min/max Zarr à partir d'un fichier brut",
    )
    build_parser.add_argument(
        "--raw",
        required=True,
        help="Chemin vers le fichier brut (EDF/FIF/...).",
    )
    build_parser.add_argument(
        "--out",
        required=True,
        help="Dossier de sortie pour la pyramide multiscale (Zarr).",
    )
    build_parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=20,
        help="Durée des chunks temporels (secondes). Défaut: 20",
    )
    build_parser.add_argument(
        "--levels",
        help="Liste de bin sizes séparés par des virgules (ex: 1,2,4,8). Défaut: puissances de deux automatiques.",
    )
    build_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Rebâtir la pyramide depuis zéro (supprime l'existant).",
    )
    build_parser.set_defaults(func=_cmd_build_pyramid)

    view_parser = subparsers.add_parser(
        "view",
        help="Lancer l'interface avec un magasin multiscale pré-calculé",
    )
    view_parser.add_argument(
        "--ms-path",
        required=True,
        help="Dossier Zarr contenant la pyramide multiscale.",
    )
    view_parser.set_defaults(func=_cmd_view)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())



