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
import subprocess
from pathlib import Path
from typing import Dict, List


def check_eval_environment() -> bool:
    """
    Check if evaluation environment exists and has required packages

    Returns:
        bool: True if environment is ready
    """
    env_name = "openmaterial_eval"

    # Check if conda env exists
    result = subprocess.run(
        f"conda env list | grep {env_name}",
        shell=True,
        capture_output=True
    )

    if result.returncode != 0:
        return False

    # Check if required packages are installed
    check_cmd = f"""
    source $(conda info --base)/etc/profile.d/conda.sh && \
    conda activate {env_name} && \
    python -c "import torch; import pytorch3d; import trimesh"
    """
    result = subprocess.run(check_cmd, shell=True, capture_output=True)

    return result.returncode == 0


def setup_eval_environment() -> bool:
    """
    Setup evaluation environment with PyTorch3D and dependencies

    Returns:
        bool: True if setup successful
    """
    env_name = "openmaterial_eval"

    print(f"\nSetting up evaluation environment: {env_name}")
    print("This may take a few minutes...")

    # Create environment
    print("Creating conda environment...")
    result = subprocess.run(
        f"conda create -n {env_name} python=3.10 -y",
        shell=True,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Failed to create environment: {result.stderr}")
        return False

    # Install PyTorch
    print("Installing PyTorch...")
    install_cmd = f"""
    source $(conda info --base)/etc/profile.d/conda.sh && \
    conda activate {env_name} && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    """
    result = subprocess.run(install_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to install PyTorch: {result.stderr}")
        return False

    # Install PyTorch3D
    print("Installing PyTorch3D...")
    pytorch3d_cmd = f"""
    source $(conda info --base)/etc/profile.d/conda.sh && \
    conda activate {env_name} && \
    pip install pytorch3d -i https://pypi.tuna.tsinghua.edu.cn/simple
    """
    result = subprocess.run(pytorch3d_cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print("PyPI installation failed, trying from source...")
        pytorch3d_source = f"""
        source $(conda info --base)/etc/profile.d/conda.sh && \
        conda activate {env_name} && \
        pip install "git+https://github.com/facebookresearch/pytorch3d.git"
        """
        result = subprocess.run(pytorch3d_source, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to install PyTorch3D: {result.stderr}")
            return False

    # Install other dependencies
    print("Installing evaluation dependencies...")
    deps_cmd = f"""
    source $(conda info --base)/etc/profile.d/conda.sh && \
    conda activate {env_name} && \
    pip install trimesh tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
    """
    result = subprocess.run(deps_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to install dependencies: {result.stderr}")
        return False

    print(f"✓ Evaluation environment '{env_name}' setup complete\n")
    return True


def load_mesh(file_path):
    """Load mesh file and convert to PyTorch3D format"""
    import torch
    import trimesh
    from pytorch3d.structures import Meshes

    mesh = trimesh.load(file_path, process=False)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
    faces = torch.tensor(mesh.faces, dtype=torch.int64).cuda()
    return Meshes(verts=[verts], faces=[faces])


def nearest_dist(pts0, pts1, batch_size=512):
    """Compute nearest distance from pts0 to pts1"""
    import torch

    pn0 = pts0.shape[0]
    dists = []
    for i in range(0, pn0, batch_size):
        dist = torch.norm(pts0[i:i+batch_size, None, :] - pts1[None, :, :], dim=-1)
        dists.append(torch.min(dist, 1)[0])
    dists = torch.cat(dists, 0)
    return dists


def compute_chamfer_distance(pred_mesh_path: str, gt_mesh_path: str,
                            num_samples: int = 1000000,
                            max_dist: float = 1.0) -> float:
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
    import torch
    from pytorch3d.ops import sample_points_from_meshes

    # Load meshes
    mesh_pr = load_mesh(pred_mesh_path)
    mesh_gt = load_mesh(gt_mesh_path)

    # Sample points
    pts_pr = sample_points_from_meshes(mesh_pr, num_samples=num_samples).squeeze()
    pts_gt = sample_points_from_meshes(mesh_gt, num_samples=num_samples).squeeze()

    # Normalize both to unit sphere
    pts_pr_center = pts_pr.mean(dim=0)
    pts_gt_center = pts_gt.mean(dim=0)
    pts_pr = pts_pr - pts_pr_center
    pts_gt = pts_gt - pts_gt_center

    pred_scale = pts_pr.abs().max()
    gt_scale = pts_gt.abs().max()
    pts_pr = pts_pr / pred_scale
    pts_gt = pts_gt / gt_scale

    # Align pred to GT using Procrustes (scale + rotation)
    # Use SVD to find optimal rotation
    H = pts_pr.T @ pts_gt  # Cross-covariance
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply rotation to pred
    pts_pr = pts_pr @ R

    # Compute bidirectional nearest distances
    dist_gt = nearest_dist(pts_gt, pts_pr)
    dist_pr = nearest_dist(pts_pr, pts_gt)

    # Filter outliers and compute mean
    dist_gt_cpu = dist_gt.cpu().numpy()
    dist_pr_cpu = dist_pr.cpu().numpy()

    mean_gt = dist_gt_cpu[dist_gt_cpu < max_dist].mean()
    mean_pr = dist_pr_cpu[dist_pr_cpu < max_dist].mean()

    # Chamfer distance in cm (matching official implementation)
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

    from tqdm import tqdm

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

            print(f"{scene_name}: {chamfer:.5f} cm")

        except Exception as e:
            print(f"Error evaluating {scene_name}: {e}")
            results['scenes'].append({
                'scene': scene_name,
                'object': object_name,
                'error': str(e)
            })

    # Compute mean (ignoring nan values)
    if results['chamfer_distances']:
        import math
        valid_distances = [d for d in results['chamfer_distances'] if not math.isnan(d)]
        if valid_distances:
            results['mean_chamfer'] = sum(valid_distances) / len(valid_distances)
        results['valid_scenes'] = len(valid_distances)
        results['nan_scenes'] = len(results['chamfer_distances']) - len(valid_distances)

    return results


def main():
    # Check and setup evaluation environment first
    print("Checking evaluation environment...")
    if not check_eval_environment():
        print("\n" + "="*60)
        print(" Evaluation Environment Not Found")
        print("="*60)
        print("\nThe evaluation requires a separate conda environment")
        print("with PyTorch3D and dependencies.")
        print("\nWould you like to set it up now? This will take a few minutes.")

        response = input("\nSetup evaluation environment? (y/n): ")
        if response.lower() != 'y':
            print("\nPlease setup the environment manually:")
            print("  conda create -n openmaterial_eval python=3.10")
            print("  conda activate openmaterial_eval")
            print("  pip install torch pytorch3d trimesh")
            sys.exit(1)

        if not setup_eval_environment():
            print("\nFailed to setup evaluation environment")
            sys.exit(1)
    else:
        print("✓ Evaluation environment ready\n")

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

    # Import torch here to check CUDA after environment is verified
    import torch

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
            print(f"  Mean Chamfer Distance: {result['mean_chamfer']:.5f} cm")

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
