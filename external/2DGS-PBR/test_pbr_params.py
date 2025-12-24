#!/usr/bin/env python3
"""
Phase 1 Test: Verify PBR parameters in GaussianModel
"""

import torch
import numpy as np
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud


def test_pbr_initialization():
    """Test PBR parameter initialization"""
    print("=" * 50)
    print("Test 1: PBR Parameter Initialization")
    print("=" * 50)

    # Create a simple point cloud
    num_points = 100
    points = np.random.randn(num_points, 3).astype(np.float32)
    colors = np.random.rand(num_points, 3).astype(np.float32)
    normals = np.zeros((num_points, 3), dtype=np.float32)

    pcd = BasicPointCloud(points=points, colors=colors, normals=normals)

    # Test without PBR
    print("\n1.1 Testing without PBR mode...")
    model_no_pbr = GaussianModel(sh_degree=3, use_pbr=False)
    model_no_pbr.create_from_pcd(pcd, spatial_lr_scale=1.0)

    assert model_no_pbr._albedo.shape[0] == 0, "Albedo should be empty without PBR"
    print("  - Without PBR: albedo empty (OK)")

    # Test with PBR
    print("\n1.2 Testing with PBR mode...")
    model_pbr = GaussianModel(sh_degree=3, use_pbr=True)
    model_pbr.create_from_pcd(pcd, spatial_lr_scale=1.0)

    assert model_pbr._albedo.shape == (num_points, 3), f"Albedo shape mismatch: {model_pbr._albedo.shape}"
    assert model_pbr._roughness.shape == (num_points, 1), f"Roughness shape mismatch: {model_pbr._roughness.shape}"
    assert model_pbr._metallic.shape == (num_points, 1), f"Metallic shape mismatch: {model_pbr._metallic.shape}"

    print(f"  - Albedo shape: {model_pbr._albedo.shape} (OK)")
    print(f"  - Roughness shape: {model_pbr._roughness.shape} (OK)")
    print(f"  - Metallic shape: {model_pbr._metallic.shape} (OK)")

    # Check value ranges after activation
    albedo = model_pbr.get_albedo
    roughness = model_pbr.get_roughness
    metallic = model_pbr.get_metallic

    print(f"\n  - Albedo range: [{albedo.min().item():.3f}, {albedo.max().item():.3f}]")
    print(f"  - Roughness range: [{roughness.min().item():.3f}, {roughness.max().item():.3f}]")
    print(f"  - Metallic range: [{metallic.min().item():.3f}, {metallic.max().item():.3f}]")

    assert 0 <= albedo.min() and albedo.max() <= 1, "Albedo should be in [0, 1]"
    assert 0.1 <= roughness.min() and roughness.max() <= 0.999, "Roughness should be in [0.1, 0.999]"
    assert 0 <= metallic.min() and metallic.max() <= 1, "Metallic should be in [0, 1]"

    print("  - Value ranges after activation: OK")

    return model_pbr


def test_pbr_save_load(model_pbr, tmp_path="/tmp/test_pbr_model.ply"):
    """Test PBR parameter save/load"""
    print("\n" + "=" * 50)
    print("Test 2: PBR Parameter Save/Load")
    print("=" * 50)

    # Save
    print(f"\n2.1 Saving to {tmp_path}...")
    model_pbr.save_ply(tmp_path)
    print("  - Save: OK")

    # Load
    print("\n2.2 Loading from file...")
    model_loaded = GaussianModel(sh_degree=3, use_pbr=True)
    model_loaded.load_ply(tmp_path)

    # Compare
    assert model_loaded._albedo.shape == model_pbr._albedo.shape, "Albedo shape mismatch after load"
    assert model_loaded._roughness.shape == model_pbr._roughness.shape, "Roughness shape mismatch after load"
    assert model_loaded._metallic.shape == model_pbr._metallic.shape, "Metallic shape mismatch after load"

    # Check values are close (may have some float precision differences)
    albedo_diff = (model_loaded._albedo - model_pbr._albedo).abs().max().item()
    roughness_diff = (model_loaded._roughness - model_pbr._roughness).abs().max().item()
    metallic_diff = (model_loaded._metallic - model_pbr._metallic).abs().max().item()

    print(f"  - Albedo max diff: {albedo_diff:.6f}")
    print(f"  - Roughness max diff: {roughness_diff:.6f}")
    print(f"  - Metallic max diff: {metallic_diff:.6f}")

    assert albedo_diff < 1e-5, f"Albedo values differ too much: {albedo_diff}"
    assert roughness_diff < 1e-5, f"Roughness values differ too much: {roughness_diff}"
    assert metallic_diff < 1e-5, f"Metallic values differ too much: {metallic_diff}"

    print("  - Load: OK")

    # Cleanup
    os.remove(tmp_path)
    print(f"  - Cleaned up {tmp_path}")


def test_pbr_optimizer():
    """Test PBR parameters in optimizer"""
    print("\n" + "=" * 50)
    print("Test 3: PBR Parameters in Optimizer")
    print("=" * 50)

    # Create model
    num_points = 50
    points = np.random.randn(num_points, 3).astype(np.float32)
    colors = np.random.rand(num_points, 3).astype(np.float32)
    normals = np.zeros((num_points, 3), dtype=np.float32)
    pcd = BasicPointCloud(points=points, colors=colors, normals=normals)

    model = GaussianModel(sh_degree=3, use_pbr=True)
    model.create_from_pcd(pcd, spatial_lr_scale=1.0)

    # Create a simple training args mock
    class TrainingArgs:
        position_lr_init = 0.00016
        position_lr_final = 0.0000016
        position_lr_delay_mult = 0.01
        position_lr_max_steps = 30000
        feature_lr = 0.0025
        opacity_lr = 0.05
        scaling_lr = 0.005
        rotation_lr = 0.001
        percent_dense = 0.01
        albedo_lr = 0.001
        roughness_lr = 0.0002
        metallic_lr = 0.0002

    print("\n3.1 Setting up training...")
    model.training_setup(TrainingArgs())

    # Check optimizer has PBR parameters
    param_names = [group["name"] for group in model.optimizer.param_groups]
    print(f"  - Optimizer param groups: {param_names}")

    assert "albedo" in param_names, "Albedo not in optimizer"
    assert "roughness" in param_names, "Roughness not in optimizer"
    assert "metallic" in param_names, "Metallic not in optimizer"

    print("  - PBR parameters in optimizer: OK")

    # Test gradient flow
    print("\n3.2 Testing gradient flow...")
    albedo = model.get_albedo
    loss = albedo.sum()
    loss.backward()

    assert model._albedo.grad is not None, "Albedo gradient is None"
    print(f"  - Albedo gradient: {model._albedo.grad.abs().mean().item():.6f}")
    print("  - Gradient flow: OK")


def main():
    print("=" * 50)
    print("2DGS-PBR Phase 1 Test: GaussianModel PBR Parameters")
    print("=" * 50)

    try:
        # Test 1: Initialization
        model = test_pbr_initialization()

        # Test 2: Save/Load
        test_pbr_save_load(model)

        # Test 3: Optimizer
        test_pbr_optimizer()

        print("\n" + "=" * 50)
        print("ALL TESTS PASSED!")
        print("=" * 50)
        return 0

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
