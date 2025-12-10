#!/usr/bin/env python3
"""
Evaluate benchmark results: compute Chamfer Distance between predicted and ground truth meshes

Usage:
    python evaluate_benchmark.py --method 2dgs --gt_dir /path/to/groundtruth --dataset_dir /path/to/dataset --output results.json
    python evaluate_benchmark.py --method all --gt_dir /path/to/groundtruth --dataset_dir /path/to/dataset --output results.json
"""

import argparse
import json
import os
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
    import os

    env_name = "openmaterial_eval"

    # Check if we're already in the correct environment
    current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if current_env == env_name:
        # Already in the right environment, just check packages
        try:
            import torch
            import pytorch3d
            import trimesh
            import cv2
            import mitsuba
            return True
        except ImportError:
            return False

    # Check if conda env exists
    result = subprocess.run(
        f"conda env list | grep {env_name}",
        shell=True,
        executable='/bin/bash',
        capture_output=True
    )

    if result.returncode != 0:
        return False

    # Check if required packages are installed
    check_cmd = f"""
    source $(conda info --base)/etc/profile.d/conda.sh && \
    conda activate {env_name} && \
    python -c "import torch; import pytorch3d; import trimesh; import cv2; import mitsuba"
    """
    result = subprocess.run(
        check_cmd,
        shell=True,
        executable='/bin/bash',
        capture_output=True
    )

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

    # Check if environment exists
    check_env = subprocess.run(
        f"conda env list | grep {env_name}",
        shell=True,
        executable='/bin/bash',
        capture_output=True
    )

    # Create environment only if it doesn't exist
    if check_env.returncode != 0:
        print("Creating conda environment...")
        result = subprocess.run(
            f"conda create -n {env_name} python=3.10 -y",
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Failed to create environment: {result.stderr}")
            return False
    else:
        print(f"✓ Conda environment '{env_name}' already exists")

    # Install PyTorch (check if already installed)
    print("Checking PyTorch...")
    check_torch = subprocess.run(
        f"""
        source $(conda info --base)/etc/profile.d/conda.sh && \
        conda activate {env_name} && \
        python -c "import torch; print(torch.__version__)" 2>/dev/null
        """,
        shell=True,
        executable='/bin/bash',
        capture_output=True,
        text=True
    )

    if check_torch.returncode == 0 and "2.3.1" in check_torch.stdout:
        print(f"✓ PyTorch 2.3.1 already installed")
    else:
        print("Installing PyTorch 2.3.1 (CUDA 12.1)...")
        install_cmd = f"""
        source $(conda info --base)/etc/profile.d/conda.sh && \
        conda activate {env_name} && \
        pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
        """
        result = subprocess.run(
            install_cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Failed to install PyTorch: {result.stderr}")
            return False

    # Install PyTorch3D (check if already installed)
    print("Checking PyTorch3D...")
    check_pytorch3d = subprocess.run(
        f"""
        source $(conda info --base)/etc/profile.d/conda.sh && \
        conda activate {env_name} && \
        python -c "import pytorch3d; print(pytorch3d.__version__)" 2>/dev/null
        """,
        shell=True,
        executable='/bin/bash',
        capture_output=True,
        text=True
    )

    if check_pytorch3d.returncode == 0:
        print(f"✓ PyTorch3D already installed")
    else:
        print("Installing PyTorch3D from source (this may take several minutes)...")
        pytorch3d_cmd = f"""
        source $(conda info --base)/etc/profile.d/conda.sh && \
        conda activate {env_name} && \
        pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git'
        """
        result = subprocess.run(
            pytorch3d_cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Failed to install PyTorch3D: {result.stderr}")
            print("\nPlease install PyTorch3D manually:")
            print("  conda activate openmaterial_eval")
            print("  pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121")
            print("  pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git'")
            return False

    # Install other dependencies
    print("Installing evaluation dependencies...")
    deps_cmd = f"""
    source $(conda info --base)/etc/profile.d/conda.sh && \
    conda activate {env_name} && \
    pip install trimesh tqdm opencv-python mitsuba -i https://pypi.tuna.tsinghua.edu.cn/simple
    """
    result = subprocess.run(
        deps_cmd,
        shell=True,
        executable='/bin/bash',
        capture_output=True,
        text=True
    )
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
                            max_dist: float = 0.15) -> float:
    """
    Compute Chamfer Distance between predicted and ground truth meshes

    Args:
        pred_mesh_path: Path to predicted mesh
        gt_mesh_path: Path to ground truth mesh
        num_samples: Number of points to sample from each mesh
        max_dist: Maximum distance threshold for filtering outliers (in meters)

    Returns:
        Chamfer distance in cm (converted from meters)
    """
    import torch
    from pytorch3d.ops import sample_points_from_meshes

    # Load meshes
    mesh_pr = load_mesh(pred_mesh_path)
    mesh_gt = load_mesh(gt_mesh_path)

    # Sample points
    pts_pr = sample_points_from_meshes(mesh_pr, num_samples=num_samples).squeeze()
    pts_gt = sample_points_from_meshes(mesh_gt, num_samples=num_samples).squeeze()

    # No normalization needed - meshes are already in the same coordinate system

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


def evaluate_method(method: str, benchmark_dir: str, gt_dir: str, dataset_dir: str = None) -> Dict:
    """
    Evaluate a single method

    Args:
        method: Method name (e.g., '2dgs', 'pgsr')
        benchmark_dir: Base benchmark output directory
        gt_dir: Ground truth directory
        dataset_dir: Dataset directory containing transforms and masks (for mesh cleaning)

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
        'mean_chamfer': 0.0,
        'by_material': {}  # Per-material statistics
    }

    # Find all generated meshes
    mesh_files = list(mesh_dir.glob("**/*.ply"))

    print(f"\n{'='*60}")
    print(f" Evaluating {method}")
    print(f" Found {len(mesh_files)} meshes")
    print(f"{'='*60}\n")

    # Clean meshes using official OpenMaterial eval script if dataset_dir is provided
    if dataset_dir:
        print("Running mesh cleaning using official OpenMaterial pipeline...")

        # Get unique object names from mesh files
        object_names = set(mesh.parent.name for mesh in mesh_files)

        for object_name in object_names:
            print(f"  Cleaning meshes for object: {object_name}")

            # Check if this object has mask files in the dataset
            object_path = Path(dataset_dir) / object_name
            has_masks = False
            if object_path.exists():
                # Check if any scene under this object has mask directory
                for scene_dir in object_path.iterdir():
                    if scene_dir.is_dir():
                        mask_dir = scene_dir / "train" / "mask"
                        if mask_dir.exists() and list(mask_dir.glob("*.png")):
                            has_masks = True
                            break

            if not has_masks:
                print(f"  ⚠ Skipping {object_name}: No masks found in dataset (cannot clean mesh)")
                # Remove this object's meshes from evaluation list
                mesh_files = [m for m in mesh_files if m.parent.name != object_name]
                continue

            # Call official clean_mesh.py with DRJIT_LIBOPTIX=0 to disable OptiX (fallback to CUDA)
            clean_cmd = f"""
            source $(conda info --base)/etc/profile.d/conda.sh && \
            conda activate openmaterial_eval && \
            cd Openmaterial-main && \
            python eval/clean_mesh.py \
                --dataset_dir {Path(dataset_dir).absolute()} \
                --groundtruth_dir {Path(gt_dir).absolute()} \
                --method {method} \
                --directory {method_dir.absolute()} \
                --object_name {object_name}
            """

            # Set environment variables for subprocess
            env = os.environ.copy()
            env['DRJIT_LIBOPTIX'] = '0'

            result = subprocess.run(
                clean_cmd,
                shell=True,
                executable='/bin/bash',
                capture_output=True,
                text=True,
                env=env
            )

            if result.returncode != 0:
                print(f"  Warning: Mesh cleaning failed for {object_name}: {result.stderr}")
            else:
                print(f"  ✓ Meshes cleaned for {object_name}")

        print("Mesh cleaning complete\n")

    from tqdm import tqdm

    for pred_mesh in tqdm(mesh_files, desc=f"Evaluating {method}"):
        # Extract scene info from path
        # Path structure: meshes/object/scene.ply
        object_name = pred_mesh.parent.name
        scene_name = pred_mesh.stem

        # Extract material from scene name (format: environment-material)
        # e.g., "symmetrical_garden_4k-plastic" -> material = "plastic"
        if '-' in scene_name:
            material = scene_name.split('-')[-1]
        else:
            material = 'unknown'

        # Find ground truth mesh
        gt_mesh = Path(gt_dir) / object_name / f"clean_{object_name}.ply"

        if not gt_mesh.exists():
            print(f"Ground truth not found for {scene_name}: {gt_mesh}")
            results['scenes'].append({
                'scene': scene_name,
                'object': object_name,
                'material': material,
                'error': 'Ground truth not found'
            })
            continue

        try:
            # Use cleaned mesh if available (from official cleaning pipeline)
            cleaned_pred_mesh = pred_mesh
            if dataset_dir:
                # Official script outputs to: {method_dir}/CleanedMesh/{object_name}/{scene_name}.ply
                cleaned_mesh_path = method_dir / "CleanedMesh" / object_name / f"{scene_name}.ply"
                if cleaned_mesh_path.exists():
                    cleaned_pred_mesh = cleaned_mesh_path
                else:
                    print(f"Warning: Cleaned mesh not found for {scene_name}, using original mesh")

            # Compute Chamfer Distance
            chamfer = compute_chamfer_distance(str(cleaned_pred_mesh), str(gt_mesh))

            results['scenes'].append({
                'scene': scene_name,
                'object': object_name,
                'material': material,
                'chamfer_distance_cm': float(chamfer),
                'pred_mesh': str(pred_mesh),
                'cleaned_pred_mesh': str(cleaned_pred_mesh) if cleaned_pred_mesh != pred_mesh else None,
                'gt_mesh': str(gt_mesh)
            })
            results['chamfer_distances'].append(float(chamfer))

            # Add to per-material statistics
            if material not in results['by_material']:
                results['by_material'][material] = {
                    'chamfer_distances': [],
                    'count': 0,
                    'mean_chamfer': 0.0
                }
            results['by_material'][material]['chamfer_distances'].append(float(chamfer))
            results['by_material'][material]['count'] += 1

            print(f"{scene_name} ({material}): {chamfer:.5f} cm")

        except Exception as e:
            print(f"Error evaluating {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            results['scenes'].append({
                'scene': scene_name,
                'object': object_name,
                'material': material,
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

    # Compute per-material means
    for material, mat_data in results['by_material'].items():
        if mat_data['chamfer_distances']:
            import math
            valid_dists = [d for d in mat_data['chamfer_distances'] if not math.isnan(d)]
            if valid_dists:
                mat_data['mean_chamfer'] = sum(valid_dists) / len(valid_dists)
                mat_data['valid_count'] = len(valid_dists)
                mat_data['nan_count'] = len(mat_data['chamfer_distances']) - len(valid_dists)

    return results


def main():
    # Check and setup evaluation environment first
    print("Checking evaluation environment...")
    if not check_eval_environment():
        print("\n" + "="*60)
        print(" Evaluation Environment Setup Required")
        print("="*60)
        print("\nThe openmaterial_eval environment exists but is missing dependencies.")
        print("Required packages: torch, pytorch3d, trimesh, opencv-python, mitsuba")
        print("\nWould you like to install missing dependencies now?")

        response = input("\nInstall dependencies? (y/n): ")
        if response.lower() != 'y':
            print("\nPlease install dependencies manually:")
            print("  conda activate openmaterial_eval")
            print("  pip install torch pytorch3d trimesh opencv-python mitsuba")
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

    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Dataset directory (for mesh cleaning with transforms and masks)')

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
        results = evaluate_method(method, args.benchmark_dir, args.gt_dir, args.dataset_dir)
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
            print(f"\n{result['method']}:")
            print(f"  Total scenes evaluated: {num_scenes}")
            print(f"  Overall Mean Chamfer Distance: {result['mean_chamfer']:.5f} cm")

            # Print per-material statistics
            if result.get('by_material'):
                print(f"\n  Per-Material Results:")
                # Sort materials for consistent output
                sorted_materials = sorted(result['by_material'].items())
                for material, mat_data in sorted_materials:
                    if 'mean_chamfer' in mat_data:
                        print(f"    {material:20s}: {mat_data['mean_chamfer']:.5f} cm ({mat_data.get('valid_count', 0)} scenes)")

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
