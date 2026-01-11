#!/usr/bin/env python3
"""
Debug script to verify coordinate system consistency between GT and 2DGS rendering.
"""

import torch
import numpy as np
import cv2
import os
from pathlib import Path

def verify_normal_consistency(depth_dir, normal_dir, sample_idx=0):
    """
    Verify GT normal and depth consistency.
    """
    print("\n" + "="*60)
    print("COORDINATE SYSTEM VERIFICATION")
    print("="*60)
    
    # Find depth and normal files
    depth_files = sorted(Path(depth_dir).glob("*.png"))
    normal_files = sorted(Path(normal_dir).glob("*.png"))
    
    if not depth_files or not normal_files:
        print(f"Error: Could not find GT files in {depth_dir} or {normal_dir}")
        return
    
    # Load sample
    depth_file = depth_files[sample_idx]
    normal_file = normal_files[sample_idx]
    
    print(f"\nAnalyzing: {depth_file.name}")
    
    # Load depth
    depth_img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
    depth_f32 = depth_img.astype(np.float32)
    
    print(f"\nDepth Statistics:")
    print(f"  Shape: {depth_f32.shape}")
    print(f"  Range: {depth_f32.min():.4f} ~ {depth_f32.max():.4f}")
    print(f"  Valid pixels (>0): {(depth_f32 > 0).sum()}")
    print(f"  Zero pixels: {(depth_f32 == 0).sum()}")
    print(f"  Has negative: {(depth_f32 < 0).sum()}")
    
    # Load normal
    normal_img = cv2.imread(str(normal_file))
    normal_img_rgb = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
    normal_f32 = normal_img_rgb.astype(np.float32) / 255.0
    normal_f32 = normal_f32 * 2.0 - 1.0  # [0,1] -> [-1,1]
    
    print(f"\nNormal Statistics:")
    print(f"  Shape: {normal_f32.shape}")
    print(f"  X range: {normal_f32[:,:,0].min():.4f} ~ {normal_f32[:,:,0].max():.4f}")
    print(f"  Y range: {normal_f32[:,:,1].min():.4f} ~ {normal_f32[:,:,1].max():.4f}")
    print(f"  Z range: {normal_f32[:,:,2].min():.4f} ~ {normal_f32[:,:,2].max():.4f}")
    
    # Check normal magnitude
    normal_mag = np.linalg.norm(normal_f32, axis=2)
    print(f"  Normal magnitude range: {normal_mag.min():.4f} ~ {normal_mag.max():.4f}")
    print(f"  Near-normalized pixels: {(np.abs(normal_mag - 1.0) < 0.1).sum()}")
    
    # Sample some pixels
    print(f"\nSample Pixel Values (y, x):")
    sample_coords = [(10, 10), (normal_f32.shape[0]//2, normal_f32.shape[1]//2), 
                     (normal_f32.shape[0]-10, normal_f32.shape[1]-10)]
    
    for y, x in sample_coords:
        if 0 <= y < normal_f32.shape[0] and 0 <= x < normal_f32.shape[1]:
            d = depth_f32[y, x]
            n = normal_f32[y, x]
            n_mag = np.linalg.norm(n)
            print(f"  ({y}, {x}): depth={d:.4f}, normal={n}, mag={n_mag:.4f}")

def check_flip_yz_effect():
    """
    Mathematically verify flip_yz effect on normal vectors.
    """
    print("\n" + "="*60)
    print("FLIP_YZ TRANSFORMATION ANALYSIS")
    print("="*60)
    
    # Test normals in OpenGL space (X right, Y up, Z back)
    test_normals = np.array([
        [1, 0, 0],    # Right (+X)
        [0, 1, 0],    # Up (+Y)
        [0, 0, 1],    # Back (+Z)
        [0, -1, 0],   # Down (-Y)
    ])
    
    flip_yz = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    
    print("\nFlip YZ matrix:")
    print(flip_yz)
    
    print("\nNormal transformations (OpenGL -> Open3D):")
    for i, n in enumerate(test_normals):
        n_flipped = n @ flip_yz.T
        print(f"  {n} -> {n_flipped}")
    
    print("\nInterpretation:")
    print("  (1,0,0): Right stays Right ✓")
    print("  (0,1,0): Up -> Down (Y flip)")
    print("  (0,0,1): Back -> Front (Z flip)")
    print("  (0,-1,0): Down -> Up (Y flip)")
    
    print("\nConclusion:")
    print("  Normals MUST be transformed with flip_yz if camera is transformed!")
    print("  Currently: GT normals are NOT transformed, but camera IS transformed.")
    print("  This causes coordinate system MISMATCH!")

def create_visualization_code():
    """
    Generate code snippet for visualizing normal vectors.
    """
    code = """
# Add this to train_pbr.py around line 307 for debugging:

if iteration == loss_scheduler.stages.stage1_end:
    # DEBUG: Visualize first normal comparison
    gt_normal_sample = gt_norm[valid_mask][:100]
    pred_normal_sample = pred_norm[valid_mask][:100]
    cosine_sample = cosine_sim[valid_mask][:100]
    
    print(f"[DEBUG NORMAL LOSS - Iteration {iteration}]")
    print(f"  GT normal mean: {gt_norm.mean(dim=(1,2))}")
    print(f"  Pred normal mean: {pred_norm.mean(dim=(1,2))}")
    print(f"  Cosine similarity: {cosine_sample.mean():.6f} ± {cosine_sample.std():.6f}")
    print(f"  Sample cosines: {cosine_sample[:10]}")
    
    # Save visualization
    gt_vis = ((gt_norm.permute(1,2,0).cpu().numpy() + 1) * 0.5 * 255).astype(np.uint8)
    pred_vis = ((pred_norm.permute(1,2,0).cpu().numpy() + 1) * 0.5 * 255).astype(np.uint8)
    cv2.imwrite(f"{dataset.model_path}/debug_gt_normal_{iteration:06d}.png", cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{dataset.model_path}/debug_pred_normal_{iteration:06d}.png", cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR))
"""
    return code

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--depth_dir", help="Path to GT depth directory")
    parser.add_argument("-n", "--normal_dir", help="Path to GT normal directory")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to analyze")
    
    args = parser.parse_args()
    
    print("\n2DGS-PBR Coordinate System Debug Tool")
    print("=" * 60)
    
    # Analysis
    check_flip_yz_effect()
    
    if args.depth_dir and args.normal_dir:
        verify_normal_consistency(args.depth_dir, args.normal_dir, args.sample)
    else:
        print("\nUsage:")
        print("  python DEBUG_COORD_SYSTEM.py -d <depth_dir> -n <normal_dir> [--sample <idx>]")
        print("\nExample:")
        print("  python DEBUG_COORD_SYSTEM.py -d data/test/depth_gt -n data/test/normal_gt")
    
    print("\n" + "="*60)
    print("VISUALIZATION CODE TO ADD:")
    print("="*60)
    print(create_visualization_code())

