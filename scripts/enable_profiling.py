"""
Script utilitaire pour activer/désactiver facilement le profiling.

Permet de configurer la télémetrie selon les besoins de diagnostic.
"""

import argparse
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.telemetry import telemetry


def enable_profiling(profile_io=True, profile_render=True, track_fps=True, csv_path=None):
    """Active le profiling avec les options spécifiées."""
    telemetry.configure(
        profile_io=profile_io,
        profile_render=profile_render,
        track_fps=track_fps,
        csv_path=csv_path
    )
    print(f"✅ Profiling activé")
    print(f"   - Profile IO: {profile_io}")
    print(f"   - Profile Render: {profile_render}")
    print(f"   - Track FPS: {track_fps}")
    if csv_path:
        print(f"   - CSV path: {csv_path}")
    else:
        print(f"   - CSV path: {telemetry._config.csv_path}")


def disable_profiling():
    """Désactive le profiling."""
    telemetry.configure(
        profile_io=False,
        profile_render=False,
        track_fps=False
    )
    print("✅ Profiling désactivé")


def show_status():
    """Affiche le statut actuel du profiling."""
    enabled = telemetry._enabled
    config = telemetry._config
    
    print("Statut du profiling:")
    print(f"   - Activé: {enabled}")
    if enabled:
        print(f"   - Profile IO: {config.profile_io}")
        print(f"   - Profile Render: {config.profile_render}")
        print(f"   - Track FPS: {config.track_fps}")
        print(f"   - CSV path: {config.csv_path}")
        print(f"   - Dataset ID: {telemetry.dataset_id}")


def main():
    parser = argparse.ArgumentParser(
        description='Active/désactive le profiling de performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Activer le profiling complet
  python scripts/enable_profiling.py --enable

  # Activer uniquement le profiling du rendu
  python scripts/enable_profiling.py --enable --no-io --fps

  # Désactiver le profiling
  python scripts/enable_profiling.py --disable

  # Afficher le statut
  python scripts/enable_profiling.py --status
        """
    )
    
    parser.add_argument('--enable', action='store_true',
                       help='Active le profiling')
    parser.add_argument('--disable', action='store_true',
                       help='Désactive le profiling')
    parser.add_argument('--status', action='store_true',
                       help='Affiche le statut actuel du profiling')
    
    parser.add_argument('--io', action='store_true', default=True,
                       help='Active le profiling IO (défaut: True si --enable)')
    parser.add_argument('--no-io', action='store_false', dest='io',
                       help='Désactive le profiling IO')
    
    parser.add_argument('--render', action='store_true', default=True,
                       help='Active le profiling du rendu (défaut: True si --enable)')
    parser.add_argument('--no-render', action='store_false', dest='render',
                       help='Désactive le profiling du rendu')
    
    parser.add_argument('--fps', action='store_true', default=True,
                       help='Active le tracking FPS (défaut: True si --enable)')
    parser.add_argument('--no-fps', action='store_false', dest='fps',
                       help='Désactive le tracking FPS')
    
    parser.add_argument('--csv', type=str, default=None,
                       help='Chemin vers le fichier CSV de télémetrie (défaut: logs/telemetry.csv)')
    
    parser.add_argument('--dataset-id', type=str, default=None,
                       help='ID du dataset pour la télémetrie')
    
    args = parser.parse_args()
    
    # Si aucun argument, afficher le statut par défaut
    if not (args.enable or args.disable or args.status):
        args.status = True
    
    if args.status:
        show_status()
        return
    
    if args.enable:
        csv_path = args.csv if args.csv else None
        enable_profiling(
            profile_io=args.io,
            profile_render=args.render,
            track_fps=args.fps,
            csv_path=csv_path
        )
        
        if args.dataset_id:
            telemetry.set_dataset_id(args.dataset_id)
            print(f"   - Dataset ID: {args.dataset_id}")
    
    elif args.disable:
        disable_profiling()
    
    # Afficher le statut final
    print()
    show_status()


if __name__ == '__main__':
    main()

