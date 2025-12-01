#!/usr/bin/env python3
"""
Setup script for external method repositories

This script sets up conda environments and compiles dependencies
for all methods (NeuS2, 2DGS, PGSR, Instant-NSR-PL).

Note: External repositories are included in the main repository.
No need to clone separately.
"""

import argparse
import subprocess
import shutil
from pathlib import Path
import sys


def setup_method(method_name):
    """Setup a specific method"""
    from methods import get_method

    print(f"\n{'='*60}")
    print(f" Setting up {method_name}")
    print(f"{'='*60}\n")

    Method = get_method(method_name)

    if method_name == 'instant-nsr-pl':
        repo_path = 'instant-nsr-pl'
    else:
        repo_path = f'external/{method_name.upper() if method_name == "2dgs" else method_name.title()}'

    method = Method(repo_path=repo_path)

    if method.setup():
        print(f"\n✓ {method_name} setup complete")
        return True
    else:
        print(f"\n✗ {method_name} setup failed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Setup conda environments for all methods',
        epilog='External repositories are already included in the project.'
    )

    parser.add_argument('--setup', type=str, nargs='+',
                        choices=['neus2', '2dgs', 'pgsr', 'instant-nsr-pl', 'all'],
                        help='Setup method environments (conda + dependencies + CUDA compilation)')

    args = parser.parse_args()

    # Check if external/ exists
    external_dir = Path.cwd() / 'external'
    if not external_dir.exists():
        print("Error: external/ directory not found!")
        print("Make sure you have cloned the repository with all its contents.")
        sys.exit(1)

    # Setup environments
    if args.setup:
        methods = args.setup
        if 'all' in methods:
            methods = ['neus2', '2dgs', 'pgsr', 'instant-nsr-pl']

        success = []
        failed = []

        for method in methods:
            if setup_method(method):
                success.append(method)
            else:
                failed.append(method)

        print("\n" + "="*60)
        print(" Setup Summary")
        print("="*60)

        if success:
            print(f"\n✓ Successfully set up: {', '.join(success)}")

        if failed:
            print(f"\n✗ Failed to set up: {', '.join(failed)}")

    # Show usage if no action specified
    else:
        parser.print_help()
        print("\n" + "="*60)
        print(" Quick Start")
        print("="*60)
        print("\n1. Setup all method environments:")
        print("   python setup_methods.py --setup all")
        print("\n2. Or setup specific methods:")
        print("   python setup_methods.py --setup neus2")
        print("   python setup_methods.py --setup neus2 2dgs")
        print("\n3. Run benchmark:")
        print("   python run_benchmark.py --method neus2 --start 0 --end 50 --gpu 0")
        print("\nNote: External repositories (NeuS2, 2DGS, PGSR) are already")
        print("included in the project. Just git pull and setup!")


if __name__ == '__main__':
    main()
