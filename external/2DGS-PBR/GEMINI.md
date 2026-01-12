# 2DGS-PBR: Static Geometry Inverse Rendering

## Project Overview
This project has evolved into a specialized **Static Geometry Inverse Rendering** pipeline based on 2D Gaussian Splatting (2DGS). Unlike standard 2DGS which optimizes geometry and appearance simultaneously, this pipeline **locks the geometry** (derived from a high-quality GT Mesh/Point Cloud) and focuses exclusively on decomposing the scene into:
1.  **PBR Materials**: Albedo, Roughness, Metallic.
2.  **Environment Lighting**: High-dynamic range environment map (Skybox).

**Core Philosophy:**
*   **Geometry First**: We assume accurate geometry is available (e.g., from LiDAR or multi-view stereo).
*   **Material Focus**: By eliminating geometric ambiguity, we achieve high-quality material separation.
*   **Skybox Synthesis**: Backgrounds are rendered via the learned environment map, enabling realistic novel view synthesis.

## Workflow

### 1. Data Preparation
You need a dataset (images + transforms) and a **dense point cloud with normals**.

If you have a Mesh (`.obj`, `.ply`, `.glb`), generate the dense point cloud using the provided utility:
```bash
python scripts/sample_mesh_to_ply.py \
    --input_mesh <path/to/mesh.obj> \
    --output_ply <path/to/dense_1M.ply> \
    --num_points 1000000
```

### 2. Static PBR Training
Run the main training script. This will:
*   Initialize 2DGS surfels on the dense points (normals aligned).
*   **Lock** XYZ and Rotation.
*   Optimize Albedo, Roughness, Metallic, and Environment Light.

```bash
python train_pbr.py \
    -s <dataset_path> \
    -m <output_path> \
    --gt_ply <dense_1M.ply> \
    --eval \
    --enable_early_stopping \
    --iterations 15000
```

**Key Arguments:**
*   `--gt_ply`: Path to the dense PLY (Required).
*   `--lambda_rgb`: Weight for standard RGB loss (Default 1.0). Controls Skybox supervision.
*   `--lambda_pbr`: Weight for PBR shading loss (Default 0.1). Controls material supervision.
*   `--enable_early_stopping`: Stop training when loss plateaus.

## Architecture & Features

### Core Scripts
*   **`train_pbr.py`**: The **Static Geometry** training engine.
    *   **Skybox Rendering**: Renders the environment map for background pixels.
    *   **Masked Optimization**: Automatically loads object masks (if available in `mask/`) to focus PBR loss on the object while letting RGB loss supervise the skybox.
    *   **Early Stopping**: Intelligent termination based on loss convergence.
*   **`scripts/sample_mesh_to_ply.py`**: Data prep tool for Poisson Disk Sampling.
*   **`scene/dataset_readers.py`**: Enhanced to auto-load `train/mask/*.png`.

### Directory Structure
*   `tests/`: Unit tests and sanity checks.
*   `scripts/archive/`: Legacy scripts (including dynamic geometry training and old open3d renderers).
*   `docs/`: Detailed technical documentation (Coordinate systems, PBR math).

### Loss Design
*   **RGB Loss (Unmasked)**: Supervises the **Environment Map** to fit the background.
*   **PBR Loss (Masked)**: Supervises **Materials** to fit the object appearance.
*   **Regularization (Masked)**: Enforces smoothness on Albedo/Roughness/Metallic.

## Status (2026-01-11)
*   **Pipeline**: Stable & Production Ready.
*   **Current Task**: Fine-tuning Skybox rendering and loss balancing.
*   **Known Issues**: Ensure input PLY has valid normals (`nx, ny, nz`).

## References
*   **Original 2DGS**: [Project Page](https://surfsplatting.github.io/)
*   **Technical Docs**: See `docs/` for deep dives.