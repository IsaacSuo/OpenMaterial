#!/usr/bin/env python3
"""
Phase 3 Test: Verify PBR Shading and Environment Lighting
"""

import torch
import sys
import os
import ast

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_pbr_utils_structure():
    """Test that pbr_utils.py has required functions and classes"""
    print("=" * 50)
    print("Test 1: PBR Utils Code Structure")
    print("=" * 50)

    pbr_path = os.path.join(os.path.dirname(__file__), 'utils', 'pbr_utils.py')

    with open(pbr_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source)

    # Find required functions and classes
    required_items = {
        'EnvironmentLight': 'class',
        'fresnel_schlick': 'function',
        'distribution_ggx': 'function',
        'geometry_smith': 'function',
        'pbr_shading': 'function',
        'pbr_shading_env': 'function',
        'screen_space_pbr_shading': 'function',
    }

    found_items = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            found_items[node.name] = 'class'
        elif isinstance(node, ast.FunctionDef):
            found_items[node.name] = 'function'

    for name, item_type in required_items.items():
        assert name in found_items, f"Missing {item_type}: {name}"
        assert found_items[name] == item_type, f"{name} should be a {item_type}"
        print(f"  - {name} ({item_type}): OK")


def test_fresnel_schlick():
    """Test Fresnel-Schlick approximation"""
    print("\n" + "=" * 50)
    print("Test 2: Fresnel-Schlick Function")
    print("=" * 50)

    from utils.pbr_utils import fresnel_schlick

    # Test basic properties
    cos_theta = torch.tensor([[0.0], [0.5], [1.0]])
    f0 = torch.tensor([[0.04, 0.04, 0.04], [0.04, 0.04, 0.04], [0.04, 0.04, 0.04]])

    result = fresnel_schlick(cos_theta, f0)

    # At cos_theta=0, F should be 1.0
    assert torch.allclose(result[0], torch.ones(3), atol=1e-5), "F(0) should be ~1.0"
    print("  - F(cos_theta=0) = 1.0: OK")

    # At cos_theta=1, F should be f0
    assert torch.allclose(result[2], f0[2], atol=1e-5), "F(1) should be f0"
    print("  - F(cos_theta=1) = f0: OK")

    # Shape should be preserved
    assert result.shape == (3, 3), f"Shape mismatch: {result.shape}"
    print("  - Output shape correct: OK")


def test_distribution_ggx():
    """Test GGX distribution function"""
    print("\n" + "=" * 50)
    print("Test 3: GGX Distribution Function")
    print("=" * 50)

    from utils.pbr_utils import distribution_ggx

    # Test at n_dot_h = 1 (perfect alignment)
    n_dot_h = torch.tensor([[1.0], [0.5], [0.0]])
    roughness = torch.tensor([[0.5], [0.5], [0.5]])

    result = distribution_ggx(n_dot_h, roughness)

    # At n_dot_h=1, distribution should be maximum
    assert result[0] > result[1], "D(1) should be > D(0.5)"
    print("  - D(n_dot_h=1) is maximum: OK")

    # All values should be positive
    assert (result >= 0).all(), "Distribution should be non-negative"
    print("  - All values non-negative: OK")


def test_geometry_smith():
    """Test Smith geometry function"""
    print("\n" + "=" * 50)
    print("Test 4: Smith Geometry Function")
    print("=" * 50)

    from utils.pbr_utils import geometry_smith

    n_dot_v = torch.tensor([[1.0], [0.5]])
    n_dot_l = torch.tensor([[1.0], [0.5]])
    roughness = torch.tensor([[0.5], [0.5]])

    result = geometry_smith(n_dot_v, n_dot_l, roughness)

    # G should be in [0, 1]
    assert (result >= 0).all() and (result <= 1).all(), "G should be in [0, 1]"
    print("  - G in [0, 1]: OK")

    # At n_dot_v=1, n_dot_l=1, G should be close to 1
    assert result[0] > 0.5, "G(1,1) should be high"
    print("  - G(n_dot_v=1, n_dot_l=1) is high: OK")


def test_pbr_shading():
    """Test full PBR shading function"""
    print("\n" + "=" * 50)
    print("Test 5: PBR Shading Function")
    print("=" * 50)

    from utils.pbr_utils import pbr_shading

    N = 10
    albedo = torch.rand(N, 3) * 0.8 + 0.1  # [0.1, 0.9]
    roughness = torch.rand(N, 1) * 0.8 + 0.1
    metallic = torch.rand(N, 1)
    normal = torch.randn(N, 3)
    normal = normal / normal.norm(dim=-1, keepdim=True)
    view_dir = torch.randn(N, 3)
    view_dir = view_dir / view_dir.norm(dim=-1, keepdim=True)
    light_dir = torch.tensor([0.5, 1.0, 0.5]).expand(N, 3)
    light_dir = light_dir / light_dir.norm(dim=-1, keepdim=True)
    light_color = torch.ones(N, 3)

    result = pbr_shading(
        albedo, roughness, metallic, normal, view_dir, light_dir, light_color
    )

    # Output shape should match input
    assert result.shape == (N, 3), f"Shape mismatch: {result.shape}"
    print("  - Output shape correct: OK")

    # Output should be non-negative
    assert (result >= 0).all(), "Shading should be non-negative"
    print("  - Output non-negative: OK")


def test_environment_light():
    """Test EnvironmentLight class"""
    print("\n" + "=" * 50)
    print("Test 6: EnvironmentLight Class")
    print("=" * 50)

    from utils.pbr_utils import EnvironmentLight

    # Test default initialization
    env_light = EnvironmentLight(resolution=64)

    # Check env_map shape
    assert env_light.env_map.shape == (3, 64, 128), f"Env map shape: {env_light.env_map.shape}"
    print(f"  - Default env map shape: {env_light.env_map.shape}: OK")

    # Test sampling
    directions = torch.randn(10, 3)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    colors = env_light.sample(directions)
    assert colors.shape == (10, 3), f"Sample output shape: {colors.shape}"
    print(f"  - Sample output shape: {colors.shape}: OK")

    # Test with 2D directions
    directions_2d = torch.randn(8, 8, 3)
    directions_2d = directions_2d / directions_2d.norm(dim=-1, keepdim=True)

    colors_2d = env_light.sample(directions_2d)
    assert colors_2d.shape == (8, 8, 3), f"2D sample output shape: {colors_2d.shape}"
    print(f"  - 2D sample output shape: {colors_2d.shape}: OK")


def test_screen_space_pbr():
    """Test screen-space PBR shading"""
    print("\n" + "=" * 50)
    print("Test 7: Screen-Space PBR Shading")
    print("=" * 50)

    from utils.pbr_utils import screen_space_pbr_shading

    H, W = 32, 32

    gbuffer_albedo = torch.rand(3, H, W)
    gbuffer_roughness = torch.rand(1, H, W) * 0.8 + 0.1
    gbuffer_metallic = torch.rand(1, H, W)
    gbuffer_normal = torch.randn(3, H, W)
    gbuffer_normal = gbuffer_normal / gbuffer_normal.norm(dim=0, keepdim=True)
    gbuffer_depth = torch.rand(1, H, W) + 0.1
    camera_center = torch.tensor([0.0, 0.0, 3.0])
    camera_transform = torch.eye(4)

    result = screen_space_pbr_shading(
        gbuffer_albedo,
        gbuffer_roughness,
        gbuffer_metallic,
        gbuffer_normal,
        gbuffer_depth,
        camera_center,
        camera_transform,
    )

    assert result.shape == (3, H, W), f"Output shape: {result.shape}"
    print(f"  - Output shape: {result.shape}: OK")

    # Check value range [0, 1]
    assert result.min() >= 0 and result.max() <= 1, "Output should be clamped to [0, 1]"
    print("  - Output in [0, 1]: OK")


def main():
    print("=" * 50)
    print("2DGS-PBR Phase 3 Test: PBR Shading")
    print("=" * 50)

    try:
        test_pbr_utils_structure()
        test_fresnel_schlick()
        test_distribution_ggx()
        test_geometry_smith()
        test_pbr_shading()
        test_environment_light()
        test_screen_space_pbr()

        print("\n" + "=" * 50)
        print("ALL PHASE 3 TESTS PASSED!")
        print("=" * 50)
        return 0

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
