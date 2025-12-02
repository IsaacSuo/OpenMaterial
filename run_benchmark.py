#!/usr/bin/env python3
"""
Unified benchmark runner for OpenMaterial dataset

Usage:
    python run_benchmark.py --method neus2 --start 0 --end 50 --gpu 0
    python run_benchmark.py --method all --start 0 --end 50 --gpus 0,1,2
"""

import argparse
import sys
from pathlib import Path
import json
from typing import List, Dict
import multiprocessing as mp
from tqdm import tqdm

# Add methods to path
sys.path.insert(0, str(Path(__file__).parent))

from methods import get_method


def parse_args():
    parser = argparse.ArgumentParser(description='Run OpenMaterial benchmark')

    parser.add_argument('--method', type=str, required=True,
                        choices=['neus2', '2dgs', 'pgsr', 'instant-nsr-pl', 'all'],
                        help='Method to run (or "all" for all methods)')

    parser.add_argument('--dataset', type=str, default='datasets/openmaterial',
                        help='Path to OpenMaterial dataset')

    parser.add_argument('--output', type=str, default='benchmark_output',
                        help='Output directory')

    parser.add_argument('--start', type=int, default=0,
                        help='Start object index')

    parser.add_argument('--end', type=int, default=-1,
                        help='End object index (-1 for all)')

    parser.add_argument('--max-scenes', type=int, default=-1,
                        help='Maximum number of scenes to process (-1 for all)')

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use (for single method)')

    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs for parallel methods (e.g., "0,1,2")')

    parser.add_argument('--skip-setup', action='store_true',
                        help='Skip environment setup')

    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training if model checkpoint already exists')

    parser.add_argument('--skip-mesh', action='store_true',
                        help='Skip mesh extraction')

    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be done without executing')

    parser.add_argument('--config', type=str, default=None,
                        help='JSON config file with method-specific parameters')

    return parser.parse_args()


def get_scenes(dataset_path: str, start: int, end: int, max_scenes: int = -1) -> List[str]:
    """Get list of scenes to process"""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Get all object directories
    objects = sorted([d for d in dataset_path.iterdir() if d.is_dir()])

    if end == -1:
        end = len(objects)

    objects = objects[start:end]

    # Get all scenes for selected objects
    scenes = []
    for obj_dir in objects:
        obj_scenes = [str(s) for s in obj_dir.iterdir() if s.is_dir()]
        scenes.extend(obj_scenes)

    # Limit total number of scenes if max_scenes is set
    if max_scenes > 0:
        scenes = scenes[:max_scenes]

    return scenes


def run_method_on_scenes(method_name: str, scenes: List[str],
                         output_dir: str, gpu_id: int,
                         skip_setup: bool, skip_train: bool, skip_mesh: bool,
                         method_config: Dict = None) -> Dict:
    """Run a method on all scenes"""

    # Initialize method
    Method = get_method(method_name)

    # Map method names to their repository paths
    repo_paths = {
        'neus2': 'external/NeuS2',
        '2dgs': 'external/2DGS',
        'pgsr': 'external/PGSR',
        'instant-nsr-pl': 'instant-nsr-pl'
    }

    repo_path = repo_paths.get(method_name)
    if not repo_path:
        raise ValueError(f"Unknown method: {method_name}")

    method = Method(repo_path=repo_path)

    # Setup environment
    if not skip_setup:
        print(f"\n{'='*60}")
        print(f" Setting up {method_name}")
        print(f"{'='*60}\n")

        if not method.setup():
            print(f"✗ Failed to setup {method_name}")
            return {'method': method_name, 'success': False, 'error': 'Setup failed'}

    # Process scenes
    results = {
        'method': method_name,
        'scenes_processed': 0,
        'scenes_success': 0,
        'scenes_failed': 0,
        'scene_results': []
    }

    print(f"\n{'='*60}")
    print(f" Processing {len(scenes)} scenes with {method_name}")
    print(f" Using GPU {gpu_id}")
    print(f"{'='*60}\n")

    method_output_dir = Path(output_dir) / method_name

    for scene_path in tqdm(scenes, desc=f"{method_name}"):
        scene_result = method.process_scene(
            input_scene=scene_path,
            output_dir=str(method_output_dir),
            gpu_id=gpu_id,
            skip_train=skip_train,
            skip_mesh=skip_mesh,
            **(method_config or {})
        )

        results['scenes_processed'] += 1
        if scene_result['success']:
            results['scenes_success'] += 1
        else:
            results['scenes_failed'] += 1

        results['scene_results'].append(scene_result)

    # Save results
    results_file = method_output_dir / 'benchmark_results.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f" {method_name} Results")
    print(f"{'='*60}")
    print(f"  Processed: {results['scenes_processed']}")
    print(f"  Success:   {results['scenes_success']}")
    print(f"  Failed:    {results['scenes_failed']}")
    print(f"  Results saved to: {results_file}")
    print()

    return results


def main():
    args = parse_args()

    # Load method config if provided
    method_config = None
    if args.config:
        with open(args.config) as f:
            method_config = json.load(f)

    # Get scenes to process
    print("Scanning dataset...")
    scenes = get_scenes(args.dataset, args.start, args.end, args.max_scenes)
    print(f"Found {len(scenes)} scenes to process")

    if args.dry_run:
        print("\nDry run - would process:")
        for scene in scenes[:10]:  # Show first 10
            print(f"  {scene}")
        if len(scenes) > 10:
            print(f"  ... and {len(scenes) - 10} more")
        return

    # Determine methods and GPUs
    if args.method == 'all':
        methods = ['instant-nsr-pl', 'neus2', '2dgs', 'pgsr']

        if args.gpus:
            gpu_ids = [int(g) for g in args.gpus.split(',')]
        else:
            gpu_ids = [args.gpu] * len(methods)

        if len(gpu_ids) < len(methods):
            # Repeat GPU IDs if not enough provided
            gpu_ids = (gpu_ids * (len(methods) // len(gpu_ids) + 1))[:len(methods)]

    else:
        methods = [args.method]
        gpu_ids = [args.gpu]

    print(f"\nMethods to run: {methods}")
    print(f"GPU assignment: {dict(zip(methods, gpu_ids))}")

    # Run methods
    if len(methods) == 1:
        # Single method - run directly
        results = run_method_on_scenes(
            methods[0], scenes, args.output,
            gpu_ids[0], args.skip_setup, args.skip_train, args.skip_mesh,
            method_config
        )

    else:
        # Multiple methods - run in parallel
        print("\nRunning methods in parallel...")

        with mp.Pool(processes=len(methods)) as pool:
            tasks = [
                pool.apply_async(
                    run_method_on_scenes,
                    (method, scenes, args.output, gpu_id,
                     args.skip_setup, args.skip_train, args.skip_mesh, method_config)
                )
                for method, gpu_id in zip(methods, gpu_ids)
            ]

            results = [task.get() for task in tasks]

    # Summary
    print("\n" + "="*60)
    print(" Benchmark Complete!")
    print("="*60)

    for result in (results if isinstance(results, list) else [results]):
        print(f"\n{result['method']}:")
        print(f"  Processed: {result['scenes_processed']}")
        print(f"  Success:   {result['scenes_success']}")
        print(f"  Failed:    {result['scenes_failed']}")

    print(f"\nResults saved to: {args.output}")
    print("\nNext steps:")
    print(f"  1. Evaluate: bash evaluate_all_methods.sh")
    print(f"  2. Compare:  python compare_methods.py")


if __name__ == '__main__':
    main()
