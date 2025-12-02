#!/usr/bin/env python3
"""
Extract mesh from already trained models

Usage:
    python extract_mesh.py --method 2dgs --model benchmark_output/2dgs/models/symmetrical_garden_4k-plastic/symmetrical_garden_4k-plastic --output meshes/test.ply
    python extract_mesh.py --method pgsr --model benchmark_output/pgsr/models/object/scene --output meshes/test.ply
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from methods import get_method


def parse_args():
    parser = argparse.ArgumentParser(description='Extract mesh from trained model')

    parser.add_argument('--method', type=str, required=True,
                        choices=['neus2', '2dgs', 'pgsr', 'instant-nsr-pl'],
                        help='Method name')

    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model directory')

    parser.add_argument('--output', type=str, required=True,
                        help='Output mesh path (.ply)')

    parser.add_argument('--data', type=str, default=None,
                        help='Path to original data (required for 2dgs/pgsr)')

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')

    parser.add_argument('--iteration', type=int, default=None,
                        help='Iteration checkpoint (default: use method default)')

    parser.add_argument('--mesh-res', type=int, default=1024,
                        help='Mesh resolution for 2DGS (default: 1024)')

    return parser.parse_args()


def main():
    args = parse_args()

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Initialize method
    Method = get_method(args.method)

    repo_paths = {
        'neus2': 'external/NeuS2',
        '2dgs': 'external/2DGS',
        'pgsr': 'external/PGSR',
        'instant-nsr-pl': 'instant-nsr-pl'
    }

    repo_path = repo_paths.get(args.method)
    if not repo_path:
        raise ValueError(f"Unknown method: {args.method}")

    method = Method(repo_path=repo_path)

    # Prepare output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare kwargs
    kwargs = {}

    if args.data:
        kwargs['data_path'] = args.data
    else:
        # Try to infer data path from model path
        # Assume structure: benchmark_output/{method}/models/{object}/{scene}
        model_path = Path(args.model)
        parts = model_path.parts

        if 'models' in parts:
            idx = parts.index('models')
            if idx + 2 < len(parts):
                object_name = parts[idx + 1]
                scene_name = parts[idx + 2]

                # Try different possible dataset locations
                possible_paths = [
                    f"datasets/openmaterial/{object_name}/{scene_name}",
                    f"/opt/data/private/dataset/OpenMaterial_ablation/{object_name}/{scene_name}",
                ]

                for data_path in possible_paths:
                    if Path(data_path).exists():
                        kwargs['data_path'] = data_path
                        print(f"Inferred data path: {data_path}")
                        break

        if 'data_path' not in kwargs and args.method in ['2dgs', 'pgsr']:
            print("Warning: Could not infer data path. Please provide --data argument.")
            print("For 2DGS and PGSR, data path is required for mesh extraction.")

    if args.iteration:
        kwargs['iterations'] = args.iteration

    if args.method == '2dgs':
        kwargs['mesh_res'] = args.mesh_res

    # Extract mesh
    print(f"\nExtracting mesh from {args.model}")
    print(f"Output: {args.output}")
    print(f"Method: {args.method}")
    print(f"Config: {kwargs}\n")

    success = method.extract_mesh(
        model_path=args.model,
        output_mesh_path=str(args.output),
        **kwargs
    )

    if success:
        print(f"\n✓ Mesh extracted successfully: {args.output}")
    else:
        print(f"\n✗ Mesh extraction failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
