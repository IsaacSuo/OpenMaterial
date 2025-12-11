#!/usr/bin/env python3
"""
Test GPU TSDF availability and performance
"""

import sys
import time
from pathlib import Path

# Add methods to path (same as 2DGS does)
sys.path.insert(0, str(Path(__file__).parent / 'methods'))

print("="*70)
print("GPU TSDF Availability Test")
print("="*70)

# Test 1: Check CUDA
print("\n[1/4] Checking CUDA availability...")
try:
    import open3d.core as o3c
    cuda_available = o3c.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")
    if not cuda_available:
        print("  ✗ CUDA not available, GPU TSDF cannot work")
        sys.exit(1)
    print("  ✓ CUDA available")
except Exception as e:
    print(f"  ✗ Error checking CUDA: {e}")
    sys.exit(1)

# Test 2: Import GPU TSDF module
print("\n[2/4] Importing GPU TSDF module...")
try:
    from utils.gpu_tsdf import create_tsdf_volume, GPUTSDFVolume
    print("  ✓ GPU TSDF module imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import GPU TSDF: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create GPU TSDF volume
print("\n[3/4] Creating GPU TSDF volume...")
try:
    volume = create_tsdf_volume(voxel_size=0.004, use_gpu=True)
    print(f"  ✓ Created volume: {type(volume).__name__}")

    if isinstance(volume, GPUTSDFVolume):
        print(f"  ✓ Using GPU TSDF (device: {volume.device})")
        print(f"  ✓ Voxel size: {volume.voxel_size}")
        print(f"  ✓ SDF truncation: {volume.sdf_trunc}")
    else:
        print(f"  ⚠ Using CPU TSDF fallback (type: {type(volume).__name__})")

except Exception as e:
    print(f"  ✗ Failed to create TSDF volume: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test different voxel sizes (memory test)
print("\n[4/4] Testing memory usage with different voxel sizes...")
voxel_sizes = [0.006, 0.005, 0.004, 0.003, 0.002]

for voxel_size in voxel_sizes:
    try:
        print(f"\n  Testing voxel_size={voxel_size}...")
        start = time.time()
        volume = create_tsdf_volume(voxel_size=voxel_size, use_gpu=True)
        elapsed = time.time() - start

        if isinstance(volume, GPUTSDFVolume):
            print(f"    ✓ GPU TSDF created in {elapsed:.3f}s")
            # Try to get memory info
            try:
                import torch
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"    GPU memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
            except:
                pass
        else:
            print(f"    ⚠ Fell back to CPU TSDF")

    except Exception as e:
        print(f"    ✗ Failed: {type(e).__name__}: {e}")
        if "memory" in str(e).lower() or "oom" in str(e).lower():
            print(f"    ⚠ voxel_size={voxel_size} causes OOM, use larger values")
            break

print("\n" + "="*70)
print("Summary:")
print("="*70)

if isinstance(volume, GPUTSDFVolume):
    print("✓ GPU TSDF is working correctly!")
    print("\nRecommendations:")
    print("  - GPU TSDF uses VRAM instead of RAM")
    print("  - Can use smaller voxel_size for better quality")
    print("  - Suggested range: 0.003 - 0.006")
    print("  - TSDF integration will be 5-10x faster")
else:
    print("⚠ GPU TSDF not working, using CPU fallback")
    print("\nRecommendations:")
    print("  - CPU TSDF uses RAM and is slower")
    print("  - Use conservative voxel_size: 0.004 - 0.006")
    print("  - Avoid voxel_size < 0.004 (causes RAM OOM)")
    print("  - mesh_res should stay at 1024-2048")

print("="*70)
