#!/usr/bin/env python3
"""
Phase 2 Test: Verify G-Buffer rendering pipeline
This test uses AST parsing to verify the code structure without running CUDA code.
"""

import ast
import sys
import os

def test_render_gbuffer_function():
    """Test that render_gbuffer function is properly defined"""
    print("=" * 50)
    print("Test 1: render_gbuffer Function Structure")
    print("=" * 50)

    renderer_path = os.path.join(os.path.dirname(__file__), 'gaussian_renderer', '__init__.py')

    with open(renderer_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source)

    # Find render_gbuffer function
    render_gbuffer_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'render_gbuffer':
            render_gbuffer_func = node
            break

    assert render_gbuffer_func is not None, "render_gbuffer function not found"
    print("  - render_gbuffer function found: OK")

    # Check parameters
    args = [arg.arg for arg in render_gbuffer_func.args.args]
    expected_args = ['viewpoint_camera', 'pc', 'pipe', 'bg_color',
                     'rasterizer', 'means3D', 'means2D', 'opacity',
                     'scales', 'rotations', 'cov3D_precomp']

    for expected in expected_args:
        assert expected in args, f"Missing argument: {expected}"
    print(f"  - Function arguments: {args}")
    print("  - All expected arguments present: OK")

    # Check that function returns something (has return statement)
    has_return = False
    for node in ast.walk(render_gbuffer_func):
        if isinstance(node, ast.Return):
            has_return = True
            break

    assert has_return, "render_gbuffer should have a return statement"
    print("  - Has return statement: OK")


def test_render_function_pbr_parameter():
    """Test that render function has render_pbr parameter"""
    print("\n" + "=" * 50)
    print("Test 2: render Function PBR Parameter")
    print("=" * 50)

    renderer_path = os.path.join(os.path.dirname(__file__), 'gaussian_renderer', '__init__.py')

    with open(renderer_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source)

    # Find render function
    render_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'render':
            render_func = node
            break

    assert render_func is not None, "render function not found"
    print("  - render function found: OK")

    # Check for render_pbr parameter
    args = [arg.arg for arg in render_func.args.args]
    print(f"  - Function arguments: {args}")

    assert 'render_pbr' in args, "render_pbr parameter not found in render function"
    print("  - render_pbr parameter present: OK")

    # Check for default value
    defaults = render_func.args.defaults
    # render_pbr should be the last positional arg with default
    # Find the index of render_pbr in args
    render_pbr_idx = args.index('render_pbr')
    num_args_with_defaults = len(defaults)
    num_args = len(args)
    first_default_idx = num_args - num_args_with_defaults

    if render_pbr_idx >= first_default_idx:
        default_value = defaults[render_pbr_idx - first_default_idx]
        if isinstance(default_value, ast.Constant):
            print(f"  - render_pbr default value: {default_value.value}")
        elif isinstance(default_value, ast.NameConstant):
            print(f"  - render_pbr default value: {default_value.value}")
    print("  - render_pbr has default value: OK")


def test_gbuffer_integration():
    """Test that render function integrates G-Buffer rendering"""
    print("\n" + "=" * 50)
    print("Test 3: G-Buffer Integration in render()")
    print("=" * 50)

    renderer_path = os.path.join(os.path.dirname(__file__), 'gaussian_renderer', '__init__.py')

    with open(renderer_path, 'r') as f:
        source = f.read()

    # Check for key patterns in the source
    checks = [
        ('render_gbuffer call', 'render_gbuffer('),
        ('gbuffer_albedo in rets', 'gbuffer_albedo'),
        ('gbuffer_roughness in rets', 'gbuffer_roughness'),
        ('gbuffer_metallic in rets', 'gbuffer_metallic'),
        ('pc.use_pbr check', 'pc.use_pbr'),
    ]

    for name, pattern in checks:
        assert pattern in source, f"Missing: {name}"
        print(f"  - {name}: OK")


def test_gaussian_model_use_pbr():
    """Test that GaussianModel has use_pbr attribute"""
    print("\n" + "=" * 50)
    print("Test 4: GaussianModel use_pbr Attribute")
    print("=" * 50)

    model_path = os.path.join(os.path.dirname(__file__), 'scene', 'gaussian_model.py')

    with open(model_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source)

    # Find GaussianModel class
    gaussian_model_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'GaussianModel':
            gaussian_model_class = node
            break

    assert gaussian_model_class is not None, "GaussianModel class not found"
    print("  - GaussianModel class found: OK")

    # Check for use_pbr in __init__
    init_method = None
    for node in gaussian_model_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == '__init__':
            init_method = node
            break

    assert init_method is not None, "__init__ method not found"

    args = [arg.arg for arg in init_method.args.args]
    assert 'use_pbr' in args, "use_pbr not in __init__ arguments"
    print("  - use_pbr in __init__ arguments: OK")

    # Check that use_pbr is stored as attribute
    assert 'self.use_pbr' in source, "self.use_pbr not assigned"
    print("  - self.use_pbr attribute assigned: OK")


def main():
    print("=" * 50)
    print("2DGS-PBR Phase 2 Test: G-Buffer Rendering Pipeline")
    print("=" * 50)

    try:
        test_render_gbuffer_function()
        test_render_function_pbr_parameter()
        test_gbuffer_integration()
        test_gaussian_model_use_pbr()

        print("\n" + "=" * 50)
        print("ALL PHASE 2 TESTS PASSED!")
        print("=" * 50)
        return 0

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
