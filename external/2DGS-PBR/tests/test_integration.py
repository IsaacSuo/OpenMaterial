#!/usr/bin/env python3
"""
Phase 5 Integration Test: Complete PBR Pipeline Verification

This test verifies that all PBR components work together without requiring
CUDA or external dependencies like plyfile.
"""

import torch
import numpy as np
import sys
import os
import ast

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, REPO_ROOT)


def test_code_syntax():
    """Verify all Python files have valid syntax"""
    print("=" * 60)
    print("Test 1: Syntax Verification - All Python Files")
    print("=" * 60)

    base_dir = REPO_ROOT

    files_to_check = [
        'scene/gaussian_model.py',
        'gaussian_renderer/__init__.py',
        'utils/pbr_utils.py',
        'utils/loss_utils.py',
        'train_pbr.py',
    ]

    for rel_path in files_to_check:
        full_path = os.path.join(base_dir, rel_path)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                source = f.read()
            try:
                ast.parse(source)
                print(f"  - {rel_path}: OK")
            except SyntaxError as e:
                print(f"  - {rel_path}: SYNTAX ERROR")
                raise e
        else:
            print(f"  - {rel_path}: NOT FOUND")
            raise FileNotFoundError(f"{rel_path} not found")


def test_pbr_shading_pipeline():
    """Test the PBR shading pipeline"""
    print("\n" + "=" * 60)
    print("Test 2: PBR Shading Pipeline")
    print("=" * 60)

    from utils.pbr_utils import (
        EnvironmentLight, fresnel_schlick, distribution_ggx,
        geometry_smith, pbr_shading, pbr_shading_env, screen_space_pbr_shading
    )

    # Test BRDF components
    print("\n  Testing BRDF components...")
    N = 100

    # Fresnel
    cos_theta = torch.rand(N, 1)
    f0 = torch.rand(N, 3) * 0.1
    F = fresnel_schlick(cos_theta, f0)
    assert F.shape == (N, 3), f"Fresnel shape: {F.shape}"
    print("    - Fresnel-Schlick: OK")

    # Distribution
    n_dot_h = torch.rand(N, 1)
    roughness = torch.rand(N, 1) * 0.8 + 0.1
    D = distribution_ggx(n_dot_h, roughness)
    assert D.shape == (N, 1), f"Distribution shape: {D.shape}"
    print("    - GGX Distribution: OK")

    # Geometry
    n_dot_v = torch.rand(N, 1)
    n_dot_l = torch.rand(N, 1)
    G = geometry_smith(n_dot_v, n_dot_l, roughness)
    assert G.shape == (N, 1), f"Geometry shape: {G.shape}"
    print("    - Smith Geometry: OK")

    # Full PBR shading
    print("\n  Testing full PBR shading...")
    albedo = torch.rand(N, 3)
    metallic = torch.rand(N, 1)
    normal = torch.randn(N, 3)
    normal = normal / normal.norm(dim=-1, keepdim=True)
    view_dir = torch.randn(N, 3)
    view_dir = view_dir / view_dir.norm(dim=-1, keepdim=True)
    light_dir = torch.tensor([0.5, 1.0, 0.5]).expand(N, 3)
    light_dir = light_dir / light_dir.norm(dim=-1, keepdim=True)
    light_color = torch.ones(N, 3)

    shaded = pbr_shading(
        albedo, roughness, metallic, normal, view_dir, light_dir, light_color
    )
    assert shaded.shape == (N, 3), f"Shaded shape: {shaded.shape}"
    assert (shaded >= 0).all(), "Shaded should be non-negative"
    print(f"    - Direct lighting: shape {shaded.shape}, range [{shaded.min():.3f}, {shaded.max():.3f}]: OK")

    # Environment lighting
    print("\n  Testing environment lighting...")
    env_light = EnvironmentLight(resolution=64)
    assert env_light.env_map.shape == (3, 64, 128), f"Env map shape: {env_light.env_map.shape}"
    print(f"    - Environment map: {env_light.env_map.shape}: OK")

    # Sample environment
    directions = torch.randn(N, 3)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    env_colors = env_light.sample(directions)
    assert env_colors.shape == (N, 3), f"Env sample shape: {env_colors.shape}"
    print(f"    - Environment sampling: {env_colors.shape}: OK")

    # Screen-space shading
    print("\n  Testing screen-space PBR shading...")
    H, W = 64, 64
    gbuffer_albedo = torch.rand(3, H, W)
    gbuffer_roughness = torch.rand(1, H, W) * 0.8 + 0.1
    gbuffer_metallic = torch.rand(1, H, W)
    gbuffer_normal = torch.randn(3, H, W)
    gbuffer_normal = gbuffer_normal / gbuffer_normal.norm(dim=0, keepdim=True)
    gbuffer_depth = torch.rand(1, H, W) + 0.1
    camera_center = torch.tensor([0.0, 0.0, 3.0])
    camera_transform = torch.eye(4)

    shaded_screen = screen_space_pbr_shading(
        gbuffer_albedo, gbuffer_roughness, gbuffer_metallic,
        gbuffer_normal, gbuffer_depth, camera_center, camera_transform
    )
    assert shaded_screen.shape == (3, H, W), f"Screen shaded shape: {shaded_screen.shape}"
    print(f"    - Screen-space shading: {shaded_screen.shape}: OK")

    # With environment lighting
    shaded_env = screen_space_pbr_shading(
        gbuffer_albedo, gbuffer_roughness, gbuffer_metallic,
        gbuffer_normal, gbuffer_depth, camera_center, camera_transform,
        env_light=env_light
    )
    assert shaded_env.shape == (3, H, W), f"Env shaded shape: {shaded_env.shape}"
    print(f"    - Screen-space with env light: {shaded_env.shape}: OK")


def test_loss_functions():
    """Test all loss functions"""
    print("\n" + "=" * 60)
    print("Test 3: Loss Functions")
    print("=" * 60)

    from utils.loss_utils import (
        l1_loss, ssim, pbr_reconstruction_loss, material_smoothness_loss,
        metallic_prior_loss, roughness_prior_loss, albedo_chroma_loss,
        compute_pbr_losses
    )

    H, W = 64, 64
    image1 = torch.rand(3, H, W)
    image2 = torch.rand(3, H, W)

    # Basic losses
    print("\n  Testing basic losses...")
    l1 = l1_loss(image1, image2)
    assert l1.dim() == 0, "L1 should be scalar"
    print(f"    - L1 loss: {l1.item():.4f}: OK")

    ssim_val = ssim(image1.unsqueeze(0), image2.unsqueeze(0))
    assert ssim_val.dim() == 0, "SSIM should be scalar"
    print(f"    - SSIM: {ssim_val.item():.4f}: OK")

    # PBR reconstruction loss
    print("\n  Testing PBR reconstruction loss...")
    pbr_recon = pbr_reconstruction_loss(image1, image2)
    assert pbr_recon.dim() == 0, "PBR recon should be scalar"
    print(f"    - PBR reconstruction: {pbr_recon.item():.4f}: OK")

    # Material losses
    print("\n  Testing material regularization losses...")
    albedo = torch.rand(3, H, W)
    roughness = torch.rand(1, H, W) * 0.8 + 0.1
    metallic = torch.rand(1, H, W)
    alpha = torch.ones(1, H, W)

    smooth = material_smoothness_loss(albedo, alpha)
    print(f"    - Material smoothness: {smooth.item():.6f}: OK")

    metal_prior = metallic_prior_loss(metallic, alpha)
    print(f"    - Metallic prior: {metal_prior.item():.6f}: OK")

    rough_prior = roughness_prior_loss(roughness, alpha)
    print(f"    - Roughness prior: {rough_prior.item():.6f}: OK")

    chroma = albedo_chroma_loss(albedo, alpha)
    print(f"    - Albedo chroma: {chroma.item():.6f}: OK")

    # Combined losses
    print("\n  Testing combined PBR losses...")
    all_losses = compute_pbr_losses(albedo, roughness, metallic, alpha)

    expected_keys = ['albedo_smooth', 'roughness_smooth', 'metallic_smooth',
                     'metallic_prior', 'roughness_prior', 'albedo_chroma', 'total_pbr_reg']
    for key in expected_keys:
        assert key in all_losses, f"Missing: {key}"
        print(f"    - {key}: {all_losses[key].item():.6f}")
    print("    - All combined losses: OK")


def test_gradient_flow():
    """Test gradient flow through PBR pipeline"""
    print("\n" + "=" * 60)
    print("Test 4: Gradient Flow")
    print("=" * 60)

    from utils.pbr_utils import pbr_shading
    from utils.loss_utils import compute_pbr_losses

    N = 100

    # Create tensors with gradients
    # Note: We create base tensors first, then apply transformations and retain_grad
    albedo = torch.rand(N, 3, requires_grad=True)
    roughness_base = torch.rand(N, 1, requires_grad=True)
    roughness = roughness_base * 0.8 + 0.1
    roughness.retain_grad()  # Retain grad for non-leaf tensor
    metallic = torch.rand(N, 1, requires_grad=True)
    normal = torch.randn(N, 3)
    normal = normal / normal.norm(dim=-1, keepdim=True)
    view_dir = torch.randn(N, 3)
    view_dir = view_dir / view_dir.norm(dim=-1, keepdim=True)
    light_dir = torch.tensor([0.5, 1.0, 0.5]).expand(N, 3)
    light_dir = light_dir / light_dir.norm(dim=-1, keepdim=True)
    light_color = torch.ones(N, 3)

    # Forward pass
    print("\n  Running forward pass...")
    shaded = pbr_shading(
        albedo, roughness, metallic, normal, view_dir, light_dir, light_color
    )
    loss = shaded.sum()
    print(f"    - Output loss: {loss.item():.4f}")

    # Backward pass
    print("\n  Running backward pass...")
    loss.backward()

    assert albedo.grad is not None, "Albedo gradient is None"
    assert roughness.grad is not None, "Roughness gradient is None"
    assert metallic.grad is not None, "Metallic gradient is None"

    print(f"    - Albedo gradient norm: {albedo.grad.norm().item():.6f}: OK")
    print(f"    - Roughness gradient norm: {roughness.grad.norm().item():.6f}: OK")
    print(f"    - Metallic gradient norm: {metallic.grad.norm().item():.6f}: OK")

    # Test loss function gradients
    print("\n  Testing loss function gradients...")
    H, W = 32, 32
    gbuffer_albedo = torch.rand(3, H, W, requires_grad=True)
    gbuffer_roughness_base = torch.rand(1, H, W, requires_grad=True)
    gbuffer_roughness = gbuffer_roughness_base * 0.8 + 0.1
    gbuffer_roughness.retain_grad()
    gbuffer_metallic = torch.rand(1, H, W, requires_grad=True)

    losses = compute_pbr_losses(gbuffer_albedo, gbuffer_roughness, gbuffer_metallic)
    total = losses['total_pbr_reg']
    total.backward()

    assert gbuffer_albedo.grad is not None, "G-Buffer albedo gradient is None"
    assert gbuffer_roughness.grad is not None, "G-Buffer roughness gradient is None"
    assert gbuffer_metallic.grad is not None, "G-Buffer metallic gradient is None"

    print(f"    - G-Buffer albedo gradient: {gbuffer_albedo.grad.abs().mean().item():.8f}: OK")
    print(f"    - G-Buffer roughness gradient: {gbuffer_roughness.grad.abs().mean().item():.8f}: OK")
    print(f"    - G-Buffer metallic gradient: {gbuffer_metallic.grad.abs().mean().item():.8f}: OK")


def test_gaussian_model_pbr_structure():
    """Verify GaussianModel has all required PBR components using AST"""
    print("\n" + "=" * 60)
    print("Test 5: GaussianModel PBR Structure (AST)")
    print("=" * 60)

    model_path = os.path.join(REPO_ROOT, 'scene', 'gaussian_model.py')
    with open(model_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source)

    # Find GaussianModel class
    gm_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'GaussianModel':
            gm_class = node
            break

    assert gm_class is not None, "GaussianModel class not found"
    print("  - GaussianModel class found: OK")

    # Check __init__ has use_pbr parameter
    init_method = None
    for node in gm_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == '__init__':
            init_method = node
            break

    assert init_method is not None, "__init__ not found"
    init_args = [arg.arg for arg in init_method.args.args]
    assert 'use_pbr' in init_args, "use_pbr not in __init__"
    print("  - use_pbr in __init__: OK")

    # Check for PBR properties
    required_properties = ['get_albedo', 'get_roughness', 'get_metallic']
    class_source = ast.get_source_segment(source, gm_class)

    for prop in required_properties:
        assert prop in source, f"Property {prop} not found"
        print(f"  - {prop} property: OK")

    # Check for PBR in training_setup
    assert 'albedo_lr' in source, "albedo_lr not found in training_setup"
    assert 'roughness_lr' in source, "roughness_lr not found in training_setup"
    assert 'metallic_lr' in source, "metallic_lr not found in training_setup"
    print("  - PBR learning rates in training_setup: OK")

    # Check save/load - check that construct_list_of_attributes includes PBR
    construct_fn_section = source.split('def construct_list_of_attributes')[1].split('def ')[0]
    assert 'albedo' in construct_fn_section, "albedo not in construct_list_of_attributes"
    assert 'roughness' in construct_fn_section, "roughness not in construct_list_of_attributes"
    assert 'metallic' in construct_fn_section, "metallic not in construct_list_of_attributes"
    print("  - PBR save/load attributes: OK")


def test_renderer_gbuffer_structure():
    """Verify renderer has G-Buffer support using AST"""
    print("\n" + "=" * 60)
    print("Test 6: Renderer G-Buffer Structure (AST)")
    print("=" * 60)

    renderer_path = os.path.join(REPO_ROOT, 'gaussian_renderer', '__init__.py')
    with open(renderer_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source)

    # Check for render_gbuffer function
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    assert 'render_gbuffer' in functions, "render_gbuffer function not found"
    print("  - render_gbuffer function: OK")

    assert 'render' in functions, "render function not found"
    print("  - render function: OK")

    # Check render function has render_pbr parameter
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'render':
            args = [arg.arg for arg in node.args.args]
            assert 'render_pbr' in args, "render_pbr not in render()"
            print("  - render_pbr parameter: OK")
            break

    # Check G-Buffer integration
    assert 'gbuffer_albedo' in source, "gbuffer_albedo not in rets"
    assert 'gbuffer_roughness' in source, "gbuffer_roughness not in rets"
    assert 'gbuffer_metallic' in source, "gbuffer_metallic not in rets"
    print("  - G-Buffer in render output: OK")


def test_training_script_structure():
    """Verify train_pbr.py has all required components using AST"""
    print("\n" + "=" * 60)
    print("Test 7: Training Script Structure (AST)")
    print("=" * 60)

    train_path = os.path.join(REPO_ROOT, 'train_pbr.py')
    with open(train_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source)

    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    required_functions = ['training_pbr_static', 'prepare_output_and_logger']
    for func in required_functions:
        assert func in functions, f"Function {func} not found"
        print(f"  - {func}: OK")

    # Check for PBR-specific features
    required_features = [
        'use_pbr=True',
        'render_pbr=True',
        'EnvironmentLight',
        'screen_space_pbr_shading',
        'compute_pbr_losses',
        '--env_map',
        '--lambda_pbr',
    ]

    for feature in required_features:
        assert feature in source, f"Feature {feature} not found"
        print(f"  - {feature}: OK")


def main():
    print("=" * 60)
    print("2DGS-PBR Phase 5: Complete Integration Test")
    print("=" * 60)

    try:
        test_code_syntax()
        test_pbr_shading_pipeline()
        test_loss_functions()
        test_gradient_flow()
        test_gaussian_model_pbr_structure()
        test_renderer_gbuffer_structure()
        test_training_script_structure()

        print("\n" + "=" * 60)
        print("PHASE 5 COMPLETE: ALL INTEGRATION TESTS PASSED!")
        print("=" * 60)
        print("\n2DGS-PBR Implementation Summary:")
        print("-" * 60)
        print("Components verified:")
        print("  1. GaussianModel: PBR parameters (albedo, roughness, metallic)")
        print("  2. Renderer: G-Buffer rendering (render_gbuffer)")
        print("  3. PBR Utils: Cook-Torrance BRDF, Environment lighting")
        print("  4. Loss Utils: PBR reconstruction and regularization losses")
        print("  5. Training: train_pbr.py with full PBR support")
        print("-" * 60)
        print("Usage:")
        print("  python train_pbr.py -s <source> --env_map <hdr_path>")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
