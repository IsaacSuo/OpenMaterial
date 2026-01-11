#!/usr/bin/env python3
"""
Proposed fixes for render_gt_open3d.py

Issue: GT normal vectors are NOT transformed by flip_yz, causing coordinate system mismatch
with 2DGS which expects view-space normals.

Current behavior:
1. mesh.vertex_normals are in world space (OpenGL: X right, Y up, Z back)
2. flip_yz transforms the CAMERA, not the geometry
3. Unlit shader outputs normals directly without view transform
4. Result: GT normals are in original world space

Expected behavior:
1. Apply flip_yz to normal vectors to match transformed camera space
2. Then normals will be in Open3D space consistent with how camera sees them
"""

import numpy as np

# ============================================================================
# PROPOSED FIX #1: Apply flip_yz to normal vectors
# ============================================================================

def fix_apply_flip_to_normals(normals):
    """
    Apply coordinate system transformation to normals.
    
    Args:
        normals: [N, 3] array of normal vectors in OpenGL space
    
    Returns:
        normals_flipped: [N, 3] array in Open3D space
    """
    # The flip_yz transformation as 3x3 matrix
    flip_yz_3x3 = np.array([
        [1,  0,  0],   # X unchanged
        [0, -1,  0],   # Y flipped
        [0,  0, -1]    # Z flipped
    ], dtype=np.float32)
    
    # Apply rotation: n' = R @ n  for column vectors
    # But normals are [N, 3] so we need [N, 3] @ [3, 3]^T = [N, 3]
    normals_flipped = normals @ flip_yz_3x3.T
    
    return normals_flipped


# ============================================================================
# PROPOSED FIX #2: Unified coordinate system without flip
# ============================================================================

def fix_remove_flip_coordinate_mismatch():
    """
    Alternative: Keep everything in OpenGL space without flip_yz.
    
    This is simpler but requires:
    1. Remove flip_yz from render_gt_open3d.py
    2. Ensure 2DGS training also uses OpenGL space consistently
    3. Adjust Open3D camera setup
    """
    # Implementation would be:
    # c2w_o3d = c2w  # Don't apply flip
    # w2c = np.linalg.inv(c2w_o3d)
    # BUT: Open3D uses different convention, so this might have other issues
    pass


# ============================================================================
# Code snippets to add to render_gt_open3d.py
# ============================================================================

CODE_FIX_NORMAL_TRANSFORM = """
# NEW CODE: Around line 163, after getting mesh_normal.vertex_normals

normals = np.asarray(mesh_normal.vertex_normals)  # [N, 3] in OpenGL space

# FIX: Apply flip_yz transformation to match camera coordinate system
# The camera is transformed with flip_yz, so normals must be too
flip_yz_3x3 = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
], dtype=np.float32)

# Transform normals: n' = n @ flip_yz^T (for row vectors)
normals_transformed = normals @ flip_yz_3x3.T  # [N, 3]

# Map [-1, 1] -> [0, 1] for color encoding
colors = (normals_transformed + 1.0) * 0.5
mesh_normal.vertex_colors = o3d.utility.Vector3dVector(colors)

print(f"[FIXED] Applied flip_yz to {len(normals)} normal vectors")
print(f"  Sample original normal: {normals[0]}")
print(f"  Sample transformed normal: {normals_transformed[0]}")
"""

CODE_VERIFICATION = """
# NEW CODE: Add verification output in render_gt function

if debug:
    # Print sample normals before and after
    print(f"[DEBUG] Sample mesh vertex normal (world space):")
    print(f"  Original: {normals[0]}")
    print(f"  After flip_yz: {normals_transformed[0]}")
    
    # After rendering, check output
    print(f"[DEBUG] Sample output normal at (10, 10):")
    img_sample = np.asarray(img_o3d)[10, 10]  # Should be in [0, 255]
    img_sample_norm = img_sample / 255.0 * 2.0 - 1.0  # Convert back to [-1, 1]
    print(f"  Rendered (encoded): {img_sample}")
    print(f"  Decoded: {img_sample_norm}")
"""

CODE_DEPTH_FIX = """
# NEW CODE: Add depth validation

depth_np = np.asarray(depth_o3d)

# Check for issues
if depth_np.min() < 0:
    print(f"[WARNING] GT depth has negative values: {depth_np.min()}")
    print(f"  This might indicate Z-axis flip issue")
    
if depth_np.max() == 0:
    print(f"[WARNING] GT depth all zero! Rendering failed?")

# After processing, verify
print(f"[INFO] GT Depth range: {depth_np.min():.4f} ~ {depth_np.max():.4f}")
print(f"       Valid pixels (>0): {(depth_np > 0).sum()}/{depth_np.size}")
"""


# ============================================================================
# Detailed change instructions
# ============================================================================

INSTRUCTIONS = """
STEP-BY-STEP FIX FOR render_gt_open3d.py
========================================

1. FIND LINE 163 (Getting mesh normals):
   Current:
   ```
   normals = np.asarray(mesh_normal.vertex_normals)
   colors = (normals + 1.0) * 0.5
   ```
   
   Replace with:
   ```
   normals = np.asarray(mesh_normal.vertex_normals)  # OpenGL space
   
   # FIX: Transform normals to match flipped camera coordinate system
   flip_yz_3x3 = np.array([
       [1,  0,  0],
       [0, -1,  0],
       [0,  0, -1]
   ], dtype=np.float32)
   normals = normals @ flip_yz_3x3.T  # Apply rotation
   
   colors = (normals + 1.0) * 0.5
   ```

2. ADD VERIFICATION (after depth saving, line ~265):
   ```
   if debug:
       # Verify normal and depth consistency
       print(f"[DEBUG] First normal at (10,10): {img_np[10,10]}")
       print(f"[DEBUG] Depth range: {depth_np.min():.4f} ~ {depth_np.max():.4f}")
   ```

3. TEST THE FIX:
   ```bash
   python scripts/render_gt_open3d.py -s <dataset_path> --debug
   ```
   Look for:
   - "Applied flip_yz to N normal vectors"
   - Sample transformed normals (should be reasonable)
   - Depth range (should be positive)

4. VERIFY IN TRAINING:
   After GT generation, check training logs:
   - m_n should increase from 0.05 to 0.3+
   - m_d should change from nan to normal values
   - Loss should stabilize


RELATED ISSUE IN train_pbr.py
=============================

The normal loss calculation (line 309) assumes GT and pred normals are in same space:
```python
cosine_sim = (pred_norm * gt_norm).sum(dim=0)
```

After the fix, this should work correctly because:
1. pred_norm comes from render_normal, which transforms view→world space
2. gt_norm will now be in the same world coordinate system (after flip_yz)

If m_n still doesn't improve, check:
- Are the normals actually normalized? (should have magnitude ≈ 1)
- Are there inverted/flipped normals due to mesh orientation?
- Is the GT normal file actually being loaded correctly?
"""


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("PROPOSED FIX #1: Apply flip_yz to normal vectors")
    print("="*70)
    print(CODE_FIX_NORMAL_TRANSFORM)
    
    print("\n" + "="*70)
    print("VERIFICATION CODE")
    print("="*70)
    print(CODE_VERIFICATION)
    
    print("\n" + "="*70)
    print("DEPTH VALIDATION")
    print("="*70)
    print(CODE_DEPTH_FIX)
    
    print("\n" + "="*70)
    print("DETAILED INSTRUCTIONS")
    print("="*70)
    print(INSTRUCTIONS)

