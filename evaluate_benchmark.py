#!/usr/bin/env python3
"""
Evaluate benchmark results: compute Chamfer Distance between predicted and ground truth meshes

Usage:
    python evaluate_benchmark.py --method 2dgs --gt_dir /path/to/groundtruth --output results.json
    python evaluate_benchmark.py --method all --gt_dir /path/to/groundtruth --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import torch
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from tqdm import tqdm


def load_mesh(file_path):
    """Load mesh file and convert to PyTorch3D format"""
    mesh = trimesh.load(file_path, process=False)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
    faces = torch.tensor(mesh.faces, dtype=torch.int64).cuda()
    return Meshes(verts=[verts], faces=[faces])


def nearest_dist(pts0, pts1, batch_size=512):
    """Compute nearest distance from pts0 to pts1"""
    pn0 = pts0.shape[0]
    dists = []
    for i in range(0, pn0, batch_size):
        dist = torch.norm(pts0[i:i+batch_size, None, :] - pts1[None, :, :], dim=-1)
        dists.append(torch.min(dist, 1)[0])
    dists = torch.cat(dists, 0)
    return dists


def compute_chamfer_distance(pred_mesh_path: str, gt_mesh_path: str,
                            num_samples: int = 1000000,
                            max_dist: float = 0.15) -> float:
    """
    Compute Chamfer Distance between predicted and ground truth meshes

    Args:
        pred_mesh_path: Path to predicted mesh
        gt_mesh_path: Path to ground truth mesh
        num_samples: Number of points to sample from each mesh
        max_dist: Maximum distance threshold for filtering outliers

    Returns:
        Chamfer distance in cm
    """
    # Load meshes
    mesh_pr = load_mesh(pred_mesh_path)
    mesh_gt = load_mesh(gt_mesh_path)

    # Sample points
    pts_pr = sample_points_from_meshes(mesh_pr, num_samples=num_samples).squeeze()
    pts_gt = sample_points_from_meshes(mesh_gt, num_samples=num_samples).squeeze()

    # Compute bidirectional nearest distances
    dist_gt = nearest_dist(pts_gt, pts_pr)
    dist_pr = nearest_dist(pts_pr, pts_gt)

    # Filter outliers and compute mean
    dist_gt_cpu = dist_gt.cpu().numpy()
    dist_pr_cpu = dist_pr.cpu().numpy()

    mean_gt = dist_gt_cpu[dist_gt_cpu < max_dist].mean()
    mean_pr = dist_pr_cpu[dist_pr_cpu < max_dist].mean()

    # Chamfer distance in cm
    chamfer = (mean_gt + mean_pr) / 2 * 100

    return chamfer


def evaluate_method(method: str, benchmark_dir: str, gt_dir: str) -> Dict:
    """
    Evaluate a single method

    Args:
        method: Method name (e.g., '2dgs', 'pgsr')
        benchmark_dir: Base benchmark output directory
        gt_dir: Ground truth directory

    Returns:
        Dict with evaluation results
    """
    method_dir = Path(benchmark_dir) / method
    mesh_dir = method_dir / "meshes"

    if not mesh_dir.exists():
        print(f"Mesh directory not found: {mesh_dir}")
        return {'method': method, 'error': 'Mesh directory not found'}

    results = {
        'method': method,
        'scenes': [],
        'chamfer_distances': [],
        'mean_chamfer': 0.0
    }

    # Find all generated meshes
    mesh_files = list(mesh_dir.glob("**/*.ply"))

    print(f"\n{'='*60}")
    print(f" Evaluating {method}")
    print(f" Found {len(mesh_files)} meshes")
    print(f"{'='*60}\n")

    for pred_mesh in tqdm(mesh_files, desc=f"Evaluating {method}"):
        # Extract scene info from path
        # Path structure: meshes/object/scene.ply
        object_name = pred_mesh.parent.name
        scene_name = pred_mesh.stem

        # Find ground truth mesh
        gt_mesh = Path(gt_dir) / object_name / f"clean_{object_name}.ply"

        if not gt_mesh.exists():
            print(f"Ground truth not found for {scene_name}: {gt_mesh}")
            results['scenes'].append({
                'scene': scene_name,
                'object': object_name,
                'error': 'Ground truth not found'
            })
            continue

        try:
            # Compute Chamfer Distance
            chamfer = compute_chamfer_distance(str(pred_mesh), str(gt_mesh))

            results['scenes'].append({
                'scene': scene_name,
                'object': object_name,
                'chamfer_distance_cm': float(chamfer),
                'pred_mesh': str(pred_mesh),
                'gt_mesh': str(gt_mesh)
            })
            results['chamfer_distances'].append(float(chamfer))

            print(f"{scene_name}: {chamfer:.4f} cm")

        except Exception as e:
            print(f"Error evaluating {scene_name}: {e}")
            results['scenes'].append({
                'scene': scene_name,
                'object': object_name,
                'error': str(e)
            })

    # Compute mean
    if results['chamfer_distances']:
        results['mean_chamfer'] = sum(results['chamfer_distances']) / len(results['chamfer_distances'])

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate benchmark results')

    parser.add_argument('--method', type=str, default='all',
                        help='Method to evaluate (2dgs, pgsr, neus2, instant-nsr-pl, or all)')

    parser.add_argument('--benchmark_dir', type=str, default='benchmark_output',
                        help='Benchmark output directory')

    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Ground truth meshes directory')

    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output JSON file for results')

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, evaluation will be slow")

    # Determine methods to evaluate
    if args.method == 'all':
        methods = ['2dgs', 'pgsr', 'neus2', 'instant-nsr-pl']
    else:
        methods = [args.method]

    # Evaluate each method
    all_results = []
    for method in methods:
        results = evaluate_method(method, args.benchmark_dir, args.gt_dir)
        all_results.append(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(" Evaluation Summary")
    print(f"{'='*60}\n")

    for result in all_results:
        if 'error' in result:
            print(f"{result['method']}: {result['error']}")
        else:
            num_scenes = len([s for s in result['scenes'] if 'chamfer_distance_cm' in s])
            print(f"{result['method']}:")
            print(f"  Scenes evaluated: {num_scenes}")
            print(f"  Mean Chamfer Distance: {result['mean_chamfer']:.4f} cm")

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
