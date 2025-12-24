#!/usr/bin/env python3
"""
Phase 4 Test: Verify PBR Loss Functions and Training Script
"""

import torch
import sys
import os
import ast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_loss_utils_structure():
    """Test that loss_utils.py has PBR loss functions"""
    print("=" * 50)
    print("Test 1: PBR Loss Functions Structure")
    print("=" * 50)

    loss_path = os.path.join(os.path.dirname(__file__), 'utils', 'loss_utils.py')

    with open(loss_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source)

    required_functions = [
        'pbr_reconstruction_loss',
        'material_smoothness_loss',
        'metallic_prior_loss',
        'roughness_prior_loss',
        'albedo_chroma_loss',
        'compute_pbr_losses',
    ]

    found_functions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            found_functions.add(node.name)

    for func in required_functions:
        assert func in found_functions, f"Missing function: {func}"
        print(f"  - {func}: OK")


def test_pbr_reconstruction_loss():
    """Test PBR reconstruction loss"""
    print("\n" + "=" * 50)
    print("Test 2: PBR Reconstruction Loss")
    print("=" * 50)

    from utils.loss_utils import pbr_reconstruction_loss

    H, W = 64, 64
    shaded = torch.rand(3, H, W)
    gt = torch.rand(3, H, W)

    loss = pbr_reconstruction_loss(shaded, gt)

    assert loss.dim() == 0, "Loss should be scalar"
    assert loss >= 0, "Loss should be non-negative"
    print(f"  - Loss value: {loss.item():.4f}")
    print("  - Loss computation: OK")

    # Test that same images give low loss
    loss_same = pbr_reconstruction_loss(gt, gt)
    assert loss_same < 0.01, f"Same image loss should be near 0, got {loss_same.item()}"
    print("  - Same image test: OK")


def test_material_smoothness_loss():
    """Test material smoothness loss"""
    print("\n" + "=" * 50)
    print("Test 3: Material Smoothness Loss")
    print("=" * 50)

    from utils.loss_utils import material_smoothness_loss

    H, W = 64, 64

    # Smooth material should have low loss
    smooth_material = torch.ones(3, H, W) * 0.5
    smooth_loss = material_smoothness_loss(smooth_material)
    print(f"  - Smooth material loss: {smooth_loss.item():.6f}")

    # Noisy material should have higher loss
    noisy_material = torch.rand(3, H, W)
    noisy_loss = material_smoothness_loss(noisy_material)
    print(f"  - Noisy material loss: {noisy_loss.item():.6f}")

    assert smooth_loss < noisy_loss, "Smooth material should have lower loss"
    print("  - Smoothness comparison: OK")

    # Test with alpha mask
    alpha = torch.ones(1, H, W)
    loss_with_alpha = material_smoothness_loss(noisy_material, alpha)
    assert loss_with_alpha >= 0, "Loss with alpha should be non-negative"
    print("  - Alpha mask support: OK")


def test_metallic_prior_loss():
    """Test metallic prior loss"""
    print("\n" + "=" * 50)
    print("Test 4: Metallic Prior Loss")
    print("=" * 50)

    from utils.loss_utils import metallic_prior_loss

    H, W = 64, 64

    # Binary metallic (all 0 or 1) should have low loss
    binary_metallic = torch.zeros(1, H, W)
    binary_loss = metallic_prior_loss(binary_metallic)
    print(f"  - Binary (0) metallic loss: {binary_loss.item():.6f}")

    binary_metallic_one = torch.ones(1, H, W)
    binary_loss_one = metallic_prior_loss(binary_metallic_one)
    print(f"  - Binary (1) metallic loss: {binary_loss_one.item():.6f}")

    # Mid-value metallic (0.5) should have higher loss
    mid_metallic = torch.ones(1, H, W) * 0.5
    mid_loss = metallic_prior_loss(mid_metallic)
    print(f"  - Mid-value metallic loss: {mid_loss.item():.6f}")

    assert mid_loss > binary_loss, "Mid-value should have higher loss than binary"
    print("  - Binary prior test: OK")


def test_roughness_prior_loss():
    """Test roughness prior loss"""
    print("\n" + "=" * 50)
    print("Test 5: Roughness Prior Loss")
    print("=" * 50)

    from utils.loss_utils import roughness_prior_loss

    H, W = 64, 64

    # Roughness at target (0.5) should have low loss
    target_roughness = torch.ones(1, H, W) * 0.5
    target_loss = roughness_prior_loss(target_roughness, target_roughness=0.5)
    print(f"  - Target roughness loss: {target_loss.item():.6f}")

    # Extreme roughness should have higher loss
    extreme_roughness = torch.ones(1, H, W) * 0.9
    extreme_loss = roughness_prior_loss(extreme_roughness, target_roughness=0.5)
    print(f"  - Extreme roughness loss: {extreme_loss.item():.6f}")

    assert target_loss < extreme_loss, "Target value should have lower loss"
    print("  - Roughness prior test: OK")


def test_compute_pbr_losses():
    """Test combined PBR losses"""
    print("\n" + "=" * 50)
    print("Test 6: Combined PBR Losses")
    print("=" * 50)

    from utils.loss_utils import compute_pbr_losses

    H, W = 64, 64
    albedo = torch.rand(3, H, W)
    roughness = torch.rand(1, H, W) * 0.8 + 0.1
    metallic = torch.rand(1, H, W)
    alpha = torch.ones(1, H, W)

    losses = compute_pbr_losses(albedo, roughness, metallic, alpha)

    expected_keys = [
        'albedo_smooth', 'roughness_smooth', 'metallic_smooth',
        'metallic_prior', 'roughness_prior', 'albedo_chroma',
        'total_pbr_reg'
    ]

    for key in expected_keys:
        assert key in losses, f"Missing loss: {key}"
        print(f"  - {key}: {losses[key].item():.6f}")

    assert losses['total_pbr_reg'] >= 0, "Total should be non-negative"
    print("  - All losses computed: OK")


def test_train_pbr_script_structure():
    """Test that train_pbr.py has required functions"""
    print("\n" + "=" * 50)
    print("Test 7: PBR Training Script Structure")
    print("=" * 50)

    train_path = os.path.join(os.path.dirname(__file__), 'train_pbr.py')

    with open(train_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source)

    required_functions = [
        'training_pbr',
        'prepare_output_and_logger',
        'training_report_pbr',
    ]

    found_functions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            found_functions.add(node.name)

    for func in required_functions:
        assert func in found_functions, f"Missing function: {func}"
        print(f"  - {func}: OK")

    # Check for PBR-specific imports
    required_imports = [
        'compute_pbr_losses',
        'pbr_reconstruction_loss',
        'EnvironmentLight',
        'screen_space_pbr_shading',
    ]

    for imp in required_imports:
        assert imp in source, f"Missing import: {imp}"
        print(f"  - Import {imp}: OK")


def test_train_pbr_arguments():
    """Test that train_pbr.py has PBR-specific arguments"""
    print("\n" + "=" * 50)
    print("Test 8: PBR Training Arguments")
    print("=" * 50)

    train_path = os.path.join(os.path.dirname(__file__), 'train_pbr.py')

    with open(train_path, 'r') as f:
        source = f.read()

    required_args = [
        '--env_map',
        '--lambda_pbr',
        '--lambda_pbr_reg',
    ]

    for arg in required_args:
        assert arg in source, f"Missing argument: {arg}"
        print(f"  - {arg}: OK")


def main():
    print("=" * 50)
    print("2DGS-PBR Phase 4 Test: Loss Functions & Training")
    print("=" * 50)

    try:
        test_loss_utils_structure()
        test_pbr_reconstruction_loss()
        test_material_smoothness_loss()
        test_metallic_prior_loss()
        test_roughness_prior_loss()
        test_compute_pbr_losses()
        test_train_pbr_script_structure()
        test_train_pbr_arguments()

        print("\n" + "=" * 50)
        print("ALL PHASE 4 TESTS PASSED!")
        print("=" * 50)
        return 0

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
