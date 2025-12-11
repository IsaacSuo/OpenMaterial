#!/usr/bin/env python3
"""
2DGS Mesh Extraction Grid Search - Stage 2
Search sdf_trunc and depth_trunc with fixed mesh_res=3072 and voxel_size=0.003
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import trimesh
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from eval_utils import clean_mesh_by_mask, clean_mesh_by_visualhull


def compute_chamfer_distance(pred_mesh_path, gt_mesh_path, num_samples=10000):
    """Compute Chamfer Distance between two meshes"""
    pred_mesh = trimesh.load(pred_mesh_path)
    gt_mesh = trimesh.load(gt_mesh_path)

    # Sample points
    pred_points = pred_mesh.sample(num_samples)
    gt_points = gt_mesh.sample(num_samples)

    # Compute distances
    from scipy.spatial import cKDTree

    # pred -> gt
    tree_gt = cKDTree(gt_points)
    dist_pred_to_gt, _ = tree_gt.query(pred_points, k=1)

    # gt -> pred
    tree_pred = cKDTree(pred_points)
    dist_gt_to_pred, _ = tree_pred.query(gt_points, k=1)

    # Chamfer distance (mean of both directions)
    chamfer_dist = (np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)) / 2

    return chamfer_dist * 100  # Convert to cm


def extract_mesh_with_params(model_path, data_path, output_mesh_path,
                             mesh_res, voxel_size, sdf_trunc, depth_trunc):
    """Extract mesh from 2DGS model with specific parameters"""

    abs_model_path = Path(model_path).absolute()
    abs_data_path = Path(data_path).absolute()
    abs_output_path = Path(output_mesh_path).absolute()

    # Create output directory
    abs_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run 2DGS mesh extraction
    cmd = f"""
    source $(conda info --base)/etc/profile.d/conda.sh && \
    conda activate surfel_splatting && \
    cd external/2DGS && \
    python render.py \
        -s {abs_data_path} \
        -m {abs_model_path} \
        --iteration 30000 \
        --skip_test \
        --skip_train \
        --mesh_res {mesh_res} \
        --voxel_size {voxel_size} \
        --sdf_trunc {sdf_trunc} \
        --depth_trunc {depth_trunc}
    """

    result = subprocess.run(
        cmd,
        shell=True,
        executable='/bin/bash',
        capture_output=False,  # Show output in real-time
        text=True
    )

    if result.returncode != 0:
        print(f"  ✗ Mesh extraction failed (return code: {result.returncode})")
        return False

    # Copy extracted mesh to output location
    train_dir = abs_model_path / "train" / "ours_30000"
    source_mesh = train_dir / "fuse_post.ply"
    if not source_mesh.exists():
        source_mesh = train_dir / "fuse.ply"

    if not source_mesh.exists():
        print(f"  ✗ Mesh not found at {train_dir}")
        return False

    import shutil
    shutil.copy(source_mesh, abs_output_path)
    return True


def clean_mesh_pipeline(raw_mesh_path, dataset_dir, object_name, scene_name, gt_dir):
    """Run official mesh cleaning pipeline"""

    raw_mesh_path = Path(raw_mesh_path)
    dataset_dir = Path(dataset_dir)
    gt_dir = Path(gt_dir)

    print(f"    Debug: raw_mesh_path = {raw_mesh_path}")
    print(f"    Debug: dataset_dir = {dataset_dir}")
    print(f"    Debug: object_name = {object_name}")
    print(f"    Debug: scene_name = {scene_name}")
    print(f"    Debug: gt_dir = {gt_dir}")

    # Paths
    scene_dir = dataset_dir / object_name / scene_name
    transforms_path = scene_dir / "transforms_train.json"
    mask_dir = scene_dir / "train" / "mask"

    print(f"    Debug: scene_dir = {scene_dir}")
    print(f"    Debug: transforms_path exists = {transforms_path.exists()}")
    print(f"    Debug: mask_dir exists = {mask_dir.exists()}")

    gt_object_dir = gt_dir / object_name
    print(f"    Debug: gt_object_dir = {gt_object_dir}")
    print(f"    Debug: gt_object_dir exists = {gt_object_dir.exists()}")

    gt_meshes = list(gt_object_dir.glob("*.ply"))
    print(f"    Debug: Found {len(gt_meshes)} GT meshes: {gt_meshes}")

    if len(gt_meshes) == 0:
        raise FileNotFoundError(f"No GT mesh found in {gt_object_dir}")

    gt_mesh_path = gt_meshes[0]
    print(f"    Debug: Using GT mesh = {gt_mesh_path}")

    # Output paths
    clean_dir = raw_mesh_path.parent / "cleaned"
    clean_dir.mkdir(exist_ok=True)

    stage1_mesh = clean_dir / f"stage1_{raw_mesh_path.name}"
    stage2_mesh = clean_dir / f"stage2_{raw_mesh_path.name}"

    # Get cut_y from GT mesh bbox
    import mitsuba as mi
    mi.set_variant('scalar_rgb')
    from mitsuba import ScalarTransform4f as T

    scene_dict = {'type': 'scene'}
    scene_dict['integrator'] = {'type': 'path', 'max_depth': 65}
    scene_dict['shape_0'] = {
        'type': 'ply',
        'id': 'Material_0001',
        'filename': str(gt_mesh_path),
        'to_world': T([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    }
    scene = mi.load_dict(scene_dict)
    bbox = scene.bbox()
    cut_y = bbox.min.y

    # Stage 1: Mask-based cleaning
    print(f"    Stage 1: Mask cleaning...")
    clean_mesh_by_mask(
        str(raw_mesh_path),
        str(stage1_mesh),
        str(transforms_path),
        str(mask_dir),
        cut_y=cut_y,
        minimal_vis=2,
        mask_dilated_size=11
    )

    # Stage 2: Visual hull cleaning
    print(f"    Stage 2: Visual hull cleaning...")
    clean_mesh_by_visualhull(
        str(stage1_mesh),
        str(stage2_mesh),
        str(transforms_path),
        str(mask_dir),
        minimal_vis=2,
        mask_dilated_size=31
    )

    return stage2_mesh, gt_mesh_path


def run_grid_search(test_scene, output_dir):
    """
    Run Stage 2 grid search

    Args:
        test_scene: dict with keys 'object_name', 'scene_name', 'model_path', 'data_path'
        output_dir: directory to save results
    """

    # Configuration - Stage 2: Search sdf_trunc and depth_trunc
    # Fixed from Stage 1
    mesh_res = 3072
    voxel_size = 0.005

    # Search ranges
    # sdf_trunc = multiplier * voxel_size
    sdf_trunc_multipliers = [3.0, 4.0, 5.0, 6.0, 8.0]
    # depth_trunc values
    depth_trunc_values = [3.0, 4.0, 5.0, 6.0, 8.0]
    # Total: 5 x 5 = 25 configurations

    dataset_dir = "/opt/data/private/dataset/OpenMaterial_ablation"
    gt_dir = "/opt/data/private/dataset/OpenMaterial_ablation/groundtruth_ablation"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    best_chamfer = float('inf')
    best_params = None

    total_configs = len(sdf_trunc_multipliers) * len(depth_trunc_values)
    current = 0

    print("="*70)
    print(f"2DGS Mesh Extraction Grid Search - Stage 2")
    print(f"Test scene: {test_scene['object_name']}/{test_scene['scene_name']}")
    print(f"Fixed: mesh_res={mesh_res}, voxel_size={voxel_size}")
    print(f"Total configurations: {total_configs}")
    print("="*70)

    for sdf_mult in sdf_trunc_multipliers:
        sdf_trunc = sdf_mult * voxel_size
        for depth_trunc in depth_trunc_values:
            current += 1
            print(f"\n[{current}/{total_configs}] Testing: sdf_trunc={sdf_trunc:.4f} (x{sdf_mult}), depth_trunc={depth_trunc}")

            # Extract mesh with these parameters
            raw_mesh_path = output_dir / f"raw_sdf{sdf_trunc:.4f}_depth{depth_trunc}.ply"

            print(f"  Extracting mesh...")
            success = extract_mesh_with_params(
                test_scene['model_path'],
                test_scene['data_path'],
                raw_mesh_path,
                mesh_res,
                voxel_size,
                sdf_trunc,
                depth_trunc
            )

            if not success:
                results.append({
                    'sdf_trunc': sdf_trunc,
                    'sdf_multiplier': sdf_mult,
                    'depth_trunc': depth_trunc,
                    'chamfer_distance_cm': None,
                    'status': 'extraction_failed'
                })
                continue

            # Clean mesh
            print(f"  Cleaning mesh...")
            try:
                cleaned_mesh_path, gt_mesh_path = clean_mesh_pipeline(
                    raw_mesh_path,
                    dataset_dir,
                    test_scene['object_name'],
                    test_scene['scene_name'],
                    gt_dir
                )
            except Exception as e:
                import traceback
                print(f"  ✗ Cleaning failed: {e}")
                print(f"  ✗ Full traceback:")
                traceback.print_exc()
                results.append({
                    'sdf_trunc': sdf_trunc,
                    'sdf_multiplier': sdf_mult,
                    'depth_trunc': depth_trunc,
                    'chamfer_distance_cm': None,
                    'status': f'cleaning_failed: {e}'
                })
                continue

            # Evaluate
            print(f"  Computing Chamfer Distance...")
            try:
                chamfer_dist = compute_chamfer_distance(cleaned_mesh_path, gt_mesh_path)
                print(f"  ✓ Chamfer Distance: {chamfer_dist:.4f} cm")

                results.append({
                    'sdf_trunc': sdf_trunc,
                    'sdf_multiplier': sdf_mult,
                    'depth_trunc': depth_trunc,
                    'chamfer_distance_cm': chamfer_dist,
                    'status': 'success'
                })

                # Track best
                if chamfer_dist < best_chamfer:
                    best_chamfer = chamfer_dist
                    best_params = {
                        'mesh_res': mesh_res,
                        'voxel_size': voxel_size,
                        'sdf_trunc': sdf_trunc,
                        'sdf_multiplier': sdf_mult,
                        'depth_trunc': depth_trunc
                    }
                    print(f"  ★ New best! Chamfer: {chamfer_dist:.4f} cm")

            except Exception as e:
                print(f"  ✗ Evaluation failed: {e}")
                results.append({
                    'sdf_trunc': sdf_trunc,
                    'sdf_multiplier': sdf_mult,
                    'depth_trunc': depth_trunc,
                    'chamfer_distance_cm': None,
                    'status': f'evaluation_failed: {e}'
                })

    # Save results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"stage2_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {csv_path}")

    # Print summary
    print(f"\nSummary:")
    print(f"  Total configs tested: {len(results)}")
    print(f"  Successful: {len([r for r in results if r['status'] == 'success'])}")
    print(f"  Failed: {len([r for r in results if r['status'] != 'success'])}")

    if best_params:
        print(f"\n★ Best Parameters:")
        print(f"  mesh_res: {best_params['mesh_res']}")
        print(f"  voxel_size: {best_params['voxel_size']}")
        print(f"  sdf_trunc: {best_params['sdf_trunc']:.4f} ({best_params['sdf_multiplier']}x voxel_size)")
        print(f"  depth_trunc: {best_params['depth_trunc']}")
        print(f"  Chamfer Distance: {best_chamfer:.4f} cm")

        # Save best params
        best_params_path = output_dir / f"best_params_stage2.json"
        with open(best_params_path, 'w') as f:
            json.dump({
                'best_params': best_params,
                'chamfer_distance_cm': best_chamfer
            }, f, indent=2)
        print(f"  Saved to: {best_params_path}")

    print("="*70)

    return df, best_params, best_chamfer


if __name__ == "__main__":
    # Test scene configuration (plastic scene)
    test_scene = {
        'object_name': '5c230ea126b943b8bc1da3f5865d5cd2',
        'scene_name': 'symmetrical_garden_4k-plastic',
        'model_path': 'benchmark_output/2dgs/models/5c230ea126b943b8bc1da3f5865d5cd2/symmetrical_garden_4k-plastic',
        'data_path': '/opt/data/private/dataset/OpenMaterial_ablation/5c230ea126b943b8bc1da3f5865d5cd2/symmetrical_garden_4k-plastic'
    }

    output_dir = "grid_search_results/2dgs_stage2"

    df, best_params, best_chamfer = run_grid_search(test_scene, output_dir)
