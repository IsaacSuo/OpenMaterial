#!/usr/bin/env python3
"""
Setup script for external method repositories

This script downloads and sets up external repositories as modules
instead of embedded git submodules.
"""

import argparse
import subprocess
import shutil
from pathlib import Path
import sys


# Repository URLs
REPOS = {
    'NeuS2': {
        'url': 'https://gh-proxy.org/https://github.com/19reborn/NeuS2.git',
        'branch': 'main',
        'recursive': True
    },
    '2DGS': {
        'url': 'https://gh-proxy.org/https://github.com/hbb1/2d-gaussian-splatting.git',
        'branch': 'main',
        'recursive': True
    },
    'PGSR': {
        'url': 'https://gh-proxy.org/https://github.com/zju3dv/PGSR.git',
        'branch': 'main',
        'recursive': False
    }
}


def run_command(cmd, cwd=None):
    """Run shell command"""
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        print(f"Error: {result.stderr}")
        return False
    return True


def clone_repo(name, info, external_dir):
    """Clone a repository"""
    repo_path = external_dir / name

    if repo_path.exists():
        print(f"✓ {name} already exists at {repo_path}")
        return True

    print(f"Cloning {name}...")

    recursive_flag = '--recursive' if info['recursive'] else ''
    cmd = f"git clone {recursive_flag} {info['url']} {name}"

    if not run_command(cmd, cwd=str(external_dir)):
        return False

    print(f"✓ {name} cloned successfully")
    return True


def move_embedded_to_external():
    """Move embedded repositories to external directory"""
    root = Path.cwd()
    external_dir = root / 'external'
    external_dir.mkdir(exist_ok=True)

    moved = False

    for name in REPOS.keys():
        embedded_path = root / name
        external_path = external_dir / name

        if embedded_path.exists() and embedded_path.is_dir():
            if external_path.exists():
                print(f"! {name} already exists in external/, skipping move")
                continue

            print(f"Moving {name} to external/...")

            # Check if it's a git repo
            if (embedded_path / '.git').exists():
                # It's a git repo, just move it
                shutil.move(str(embedded_path), str(external_path))
                print(f"✓ Moved {name} to external/")
                moved = True
            else:
                print(f"! {name} is not a git repository, skipping")

    return moved


def create_data_converter():
    """Create OpenMaterial data converter for NeuS2"""
    external_dir = Path.cwd() / 'external'
    neus2_path = external_dir / 'NeuS2'

    if not neus2_path.exists():
        print("NeuS2 not found, skipping converter creation")
        return

    tools_dir = neus2_path / 'tools'
    tools_dir.mkdir(exist_ok=True)

    converter_path = tools_dir / 'convert_openmaterial.py'

    if converter_path.exists():
        print(f"✓ Converter already exists at {converter_path}")
        return

    # Copy converter from NeuS2/tools/ if we created it earlier
    old_converter = Path('NeuS2/tools/convert_openmaterial.py')
    if old_converter.exists():
        shutil.copy(old_converter, converter_path)
        print(f"✓ Copied converter to {converter_path}")
    else:
        print("! Converter script not found, you may need to create it")


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
    parser = argparse.ArgumentParser(description='Setup external method repositories')

    parser.add_argument('--clone', action='store_true',
                        help='Clone external repositories')

    parser.add_argument('--move', action='store_true',
                        help='Move embedded repositories to external/')

    parser.add_argument('--setup', type=str, nargs='+',
                        choices=['neus2', '2dgs', 'pgsr', 'instant-nsr-pl', 'all'],
                        help='Setup method environments')

    parser.add_argument('--clean', action='store_true',
                        help='Remove external repositories (dangerous!)')

    args = parser.parse_args()

    root = Path.cwd()
    external_dir = root / 'external'

    # Move embedded repos
    if args.move:
        print("\n" + "="*60)
        print(" Moving Embedded Repositories")
        print("="*60 + "\n")

        if move_embedded_to_external():
            print("\n✓ Repositories moved to external/")
            print("\nYou can now safely remove old embedded directories:")
            print("  rm -rf NeuS2/ 2DGS/ PGSR/")
        else:
            print("\nNo repositories to move")

    # Clone repositories
    if args.clone:
        print("\n" + "="*60)
        print(" Cloning External Repositories")
        print("="*60 + "\n")

        external_dir.mkdir(exist_ok=True)

        for name, info in REPOS.items():
            clone_repo(name, info, external_dir)

        # Create data converter
        create_data_converter()

        print("\n✓ All repositories cloned")

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

    # Clean
    if args.clean:
        print("\n" + "="*60)
        print(" Cleaning External Repositories")
        print("="*60 + "\n")

        confirm = input("This will delete external/. Are you sure? (yes/no): ")
        if confirm.lower() == 'yes':
            if external_dir.exists():
                shutil.rmtree(external_dir)
                print("✓ Removed external/")
            else:
                print("external/ does not exist")
        else:
            print("Cancelled")

    # Show usage if no action specified
    if not any([args.clone, args.move, args.setup, args.clean]):
        parser.print_help()
        print("\n" + "="*60)
        print(" Quick Start")
        print("="*60)
        print("\n1. Move embedded repos (if you have them):")
        print("   python setup_methods.py --move")
        print("\n2. Or clone fresh repos:")
        print("   python setup_methods.py --clone")
        print("\n3. Setup environments:")
        print("   python setup_methods.py --setup all")
        print("\n4. Run benchmark:")
        print("   python run_benchmark.py --method neus2 --start 0 --end 50 --gpu 0")


if __name__ == '__main__':
    main()
