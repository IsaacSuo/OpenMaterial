# 2DGS-PBR: 2D Gaussian Splatting with Physically Based Rendering

## Project Overview
This project is an extension of **2D Gaussian Splatting (2DGS)** that integrates **Physically Based Rendering (PBR)** capabilities. It specifically implements a learnable environment map (`EnvironmentLight`) to model scene lighting and material interactions, enhancing the geometric accuracy and visual fidelity of the reconstructed radiance fields.

**Key Features:**
*   **Learnable Environment Lighting:** Optimizes an environment map (`EnvironmentLight`) alongside Gaussian parameters.
*   **PBR Shading:** Incorporates diffuse and specular shading components using the learned environment lighting.
*   **Hybrid Optimization:** PBR optimization (loss and env map updates) activates after a set number of iterations (default: 5000) to ensure stable geometric initialization.
*   **Base 2DGS Support:** Retains all original 2DGS functionality (bounded/unbounded mesh extraction, surfel regularization).

## Directory Structure

### Core PBR Files
*   **`train_pbr.py`**: The main training script for PBR-enhanced 2DGS. It handles the dual optimization loop (Gaussians + Environment Light).
*   **`render_pbr.py`**: The rendering script for generating PBR-shaded images and evaluating the trained models.
*   **`utils/pbr_utils.py`**: Contains the `EnvironmentLight` class and PBR-related utility functions (sampling, shading logic).
*   **`PBR_*.md/txt`**: Documentation specific to the PBR implementation status and verification (e.g., `PBR_FINAL_SUMMARY.md`).

### Base 2DGS Files
*   **`train.py` / `render.py`**: Original 2DGS training and rendering scripts.
*   **`scene/`**: Handles dataset loading (COLMAP, NeRF Synthetic) and the Gaussian Model definition.
*   **`gaussian_renderer/`**: Contains the rasterization logic (integrating `diff-surfel-rasterization`).
*   **`submodules/`**: CUDA extensions for efficient rasterization (`diff-surfel-rasterization`, `simple-knn`).

## Setup and Usage

### 1. Environment Setup
The project relies on a Conda environment with PyTorch and custom CUDA extensions.

```bash
# Create the environment
conda env create --file environment.yml

# Activate
conda activate surfel_splatting
```

**Note:** Ensure CUDA toolkit is installed and compatible with the PyTorch version specified in `environment.yml` (likely 11.8 or 12.1).

### 2. Training
To train a scene with PBR features enabled:

```bash
# Standard PBR training
python train_pbr.py -s <path_to_dataset> -m <output_directory>
```

**Key Parameters:**
*   `-s, --source_path`: Path to the dataset (COLMAP or NeRF Synthetic format).
*   `-m, --model_path`: Directory to save results.
*   PBR optimization typically starts after **5000 iterations**.

### 3. Rendering & Evaluation
To render a trained PBR model:

```bash
# Render PBR output
python render_pbr.py -m <path_to_trained_model> -s <path_to_dataset>
```

The script will automatically look for `env_light_{iteration}.pth` in the model directory to load the trained lighting environment.

### 4. Base 2DGS Operations
You can still perform standard 2DGS operations using the base scripts:

*   **Standard Training:** `python train.py -s <data_path>`
*   **Mesh Extraction:** `python render.py -m <model_path> --unbounded --mesh_res 1024`

## Development Conventions

*   **PBR Implementation:** The `EnvironmentLight` class (in `utils/pbr_utils.py`) is an `nn.Module` managing the environment map as an `nn.Parameter`.
*   **Gradient Flow:** The implementation ensures a complete gradient path from the PBR loss -> Shading -> Environment Sample -> Environment Map, allowing for end-to-end optimization.
*   **Data Format:** Follows standard 3DGS/2DGS data conventions (COLMAP or NeRF Synthetic).
*   **Output:** Training results are saved in `output/` (or specified model path), including checkpoints (`chkpnt_*.pth`) and environment light weights (`env_light_*.pth`).

## References
*   **Original 2DGS:** [Project Page](https://surfsplatting.github.io/) | [Paper](https://arxiv.org/abs/2403.17888)
*   **PBR Implementation Status:** See `PBR_FINAL_SUMMARY.md` for a detailed verification report.
