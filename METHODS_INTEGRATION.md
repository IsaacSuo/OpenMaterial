# OpenMaterial Methods Integration Guide

This document provides a comprehensive guide for running NeuS2, 2DGS, and PGSR on the OpenMaterial dataset for benchmarking neural surface reconstruction methods.

## Overview

This repository now supports the following methods:
1. **Instant-NSR-PL** (baseline, already implemented)
2. **NeuS2** - Fast neural implicit surface reconstruction
3. **2DGS** - 2D Gaussian Splatting for geometrically accurate radiance fields
4. **PGSR** - Planar-based Gaussian Splatting for surface reconstruction

## Quick Start

### 1. Download OpenMaterial Dataset

```bash
# Download ablation dataset (for quick testing)
python download.py --token YOUR_HF_TOKEN --type ablation

# Or download complete dataset
python download.py --token YOUR_HF_TOKEN --type all
```

### 2. Setup Environments

#### NeuS2
```bash
cd NeuS2
conda create -n neus2 python=3.9
conda activate neus2
pip install -r requirements.txt
pip install torch torchvision pytorch3d

# Build CUDA code (requires CMake >= 3.18)
cmake . -B build
cmake --build build --config RelWithDebInfo -j
cd ..
```

#### 2DGS
```bash
cd 2DGS
conda env create --file environment.yml
conda activate surfel_splatting
# Or install in existing 3DGS environment
pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn
cd ..
```

#### PGSR
```bash
cd PGSR
conda create -n pgsr python=3.8
conda activate pgsr
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install submodules/diff-plane-rasterization
pip install submodules/simple-knn
cd ..
```

### 3. Run Methods

#### Single Method
```bash
# NeuS2
bash run_neus2_openmaterial.sh 0 50 0

# 2DGS
bash run_2dgs_openmaterial.sh 0 50 0

# PGSR
bash run_pgsr_openmaterial.sh 0 50 0
```

#### Multiple GPUs (Parallel)
```bash
# Run different methods on different GPUs
bash run_neus2_openmaterial.sh 0 50 0 &
bash run_2dgs_openmaterial.sh 0 50 1 &
bash run_pgsr_openmaterial.sh 0 50 2 &
wait
```

### 4. Evaluate Results

```bash
# Evaluate all methods
bash evaluate_all_methods.sh

# Compare methods
python compare_methods.py --methods instant-nsr-pl-wmask neus2 2dgs pgsr
```

## Detailed Usage

### NeuS2

NeuS2 requires data format conversion from OpenMaterial to NeuS2's format.

**Data Format:**
- Input: OpenMaterial format (COLMAP-like + transforms.json)
- Output: NeuS2 format (RGBA images + NeuS2 transforms.json)

**Key Features:**
- Training time: ~5-10 min per scene
- Memory: ~8-12GB
- Supports masks for better reconstruction
- Best for: General surface reconstruction

**Configuration:**
```bash
# Edit NeuS2/configs/nerf/dtu.json
{
  "n_steps": 15000,          # Training iterations
  "batch_size": 1<<18,       # Rays per batch
  "learning_rate": 0.01      # Learning rate
}
```

**Manual Usage:**
```bash
# 1. Convert data
python NeuS2/tools/convert_openmaterial.py \
    --input datasets/openmaterial/OBJECT_ID/SCENE_NAME \
    --output NeuS2_data/OBJECT_ID/SCENE_NAME

# 2. Train
cd NeuS2
python scripts/run.py \
    --scene ../NeuS2_data/OBJECT_ID/SCENE_NAME/transforms_train.json \
    --name experiment_name \
    --network dtu.json \
    --n_steps 15000 \
    --save_mesh \
    --save_mesh_path ../output/mesh.ply
```

### 2DGS

2DGS can directly use OpenMaterial's COLMAP-compatible format.

**Data Format:**
- Input: OpenMaterial format (directly compatible)
- No conversion needed!

**Key Features:**
- Training time: ~15-20 min per scene
- Memory: ~10-14GB
- Excellent mesh quality
- Best for: High-quality geometry reconstruction

**Configuration:**
```bash
# Key training parameters
python train.py \
    -s SCENE_PATH \
    --iterations 30000 \
    --lambda_normal 0.05 \      # Normal consistency regularization
    --lambda_distortion 1000 \  # Depth distortion regularization
    --depth_ratio 0             # 0 for mean depth, 1 for median
```

**Manual Usage:**
```bash
cd 2DGS

# Train
python train.py \
    -s ../datasets/openmaterial/OBJECT_ID/SCENE_NAME \
    -m ../output/2dgs/OBJECT_ID/SCENE_NAME \
    --iterations 30000

# Extract mesh
python render.py \
    -s ../datasets/openmaterial/OBJECT_ID/SCENE_NAME \
    -m ../output/2dgs/OBJECT_ID/SCENE_NAME \
    --skip_test --skip_train \
    --mesh_res 1024
```

### PGSR

PGSR also uses COLMAP-compatible format.

**Data Format:**
- Input: OpenMaterial format (directly compatible)
- No conversion needed!

**Key Features:**
- Training time: ~20-30 min per scene
- Memory: ~12-16GB
- Planar-based representation
- Best for: High-fidelity surfaces with planar regions

**Configuration:**
```bash
# Key training parameters
python train.py \
    -s SCENE_PATH \
    --iterations 30000 \
    --gpu GPU_ID
```

**Manual Usage:**
```bash
cd PGSR

# Train
python train.py \
    -s ../datasets/openmaterial/OBJECT_ID/SCENE_NAME \
    -m ../output/pgsr/OBJECT_ID/SCENE_NAME \
    --iterations 30000

# Extract mesh
python render.py \
    -s ../datasets/openmaterial/OBJECT_ID/SCENE_NAME \
    -m ../output/pgsr/OBJECT_ID/SCENE_NAME \
    --iteration 30000 \
    --compute_mesh
```

## Output Structure

```
OpenMaterial/
├── datasets/
│   └── openmaterial/          # Input data
├── NeuS2/                     # NeuS2 repository
├── 2DGS/                      # 2DGS repository
├── PGSR/                      # PGSR repository
├── NeuS2_data/                # Converted data for NeuS2
├── Mesh/                      # Output meshes
│   ├── neus2/meshes/
│   ├── 2dgs/meshes/
│   └── pgsr/meshes/
├── neus2_output/              # NeuS2 results
├── 2dgs_output/               # 2DGS results
├── pgsr_output/               # PGSR results
└── method_comparison.csv      # Comparison results
```

## Evaluation Metrics

### Novel View Synthesis
- **PSNR** (Peak Signal-to-Noise Ratio) ↑
- **SSIM** (Structural Similarity) ↑
- **LPIPS** (Learned Perceptual Image Patch Similarity) ↓

### Geometry Reconstruction
- **Chamfer Distance** ↓ (multiplied by 100)

## Troubleshooting

### NeuS2

**Problem:** CMake version too old
```bash
pip install cmake --upgrade
```

**Problem:** CUDA not found
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

**Problem:** Out of memory
- Reduce `marching_cubes_res` to 256
- Reduce batch size in config

### 2DGS

**Problem:** Submodule not found
```bash
cd 2DGS
git submodule update --init --recursive
```

**Problem:** CUDA compilation error
```bash
# Make sure CUDA toolkit matches PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
```

**Problem:** Poor mesh quality
- Increase iterations to 40000
- Adjust `lambda_normal` (try 0.1)
- Adjust `lambda_distortion` (try 500 or 2000)

### PGSR

**Problem:** Installation fails
```bash
# Install dependencies one by one
pip install torch torchvision torchaudio
pip install submodules/diff-plane-rasterization
pip install submodules/simple-knn
```

**Problem:** Training crashes
- Reduce iterations if on limited GPU
- Check if masks are properly loaded

## Performance Comparison

Expected performance on OpenMaterial dataset (RTX 3090):

| Method | Training Time | Memory | PSNR (avg) | Chamfer Distance |
|--------|--------------|--------|------------|------------------|
| Instant-NSR-PL | 5-10 min | 8GB | ~30 | ~1.5 |
| NeuS2 | 5-10 min | 10GB | ~30 | ~1.3 |
| 2DGS | 15-20 min | 12GB | ~32 | ~0.8 |
| PGSR | 20-30 min | 14GB | ~31 | ~0.7 |

*Note: These are rough estimates. Actual performance varies by material type.*

## Material-Specific Notes

### Easy Materials
- **Diffuse**: All methods perform well
- **Rough Plastic**: Good results across all methods

### Challenging Materials
- **Conductor** (high specular): 2DGS and PGSR perform better
- **Dielectric** (transparent): NeuS2 struggles, 2DGS recommended
- **Rough variants**: Generally easier than smooth counterparts

## Best Practices

1. **For Quick Testing**: Use ablation dataset first
2. **For Best Quality**: Use 2DGS with high iterations
3. **For Speed**: Use NeuS2
4. **For Planar Scenes**: Use PGSR
5. **Always**: Check GPU memory before starting large-scale experiments

## Citation

If you use these methods, please cite:

### NeuS2
```bibtex
@inproceedings{neus2,
  title={NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-view Reconstruction},
  author={Wang, Yiming and Han, Qin and Habermann, Marc and Daniilidis, Kostas and Theobalt, Christian and Liu, Lingjie},
  booktitle={ICCV},
  year={2023}
}
```

### 2DGS
```bibtex
@article{2dgs,
  title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
  author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
  journal={SIGGRAPH},
  year={2024}
}
```

### PGSR
```bibtex
@article{pgsr,
  title={PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction},
  author={Chen, Danpeng and Li, Hai and Ye, Weicai and Wang, Yifan and Xie, Weijian and Zhai, Shangjin and Wang, Nan and Liu, Haomin and Bao, Hujun and Zhang, Guofeng},
  journal={TVCG},
  year={2024}
}
```

### OpenMaterial Dataset
```bibtex
@article{openmaterial,
  title={OpenMaterial: A Comprehensive Dataset of Complex Materials for 3D Reconstruction},
  author={...},
  journal={arXiv},
  year={2024}
}
```

## Support

For issues related to:
- **This integration**: Open an issue in this repository
- **NeuS2**: Visit https://github.com/19reborn/NeuS2
- **2DGS**: Visit https://github.com/hbb1/2d-gaussian-splatting
- **PGSR**: Visit https://github.com/zju3dv/PGSR
- **OpenMaterial dataset**: Visit the official project page

## License

- This integration code: Same as OpenMaterial repository
- NeuS2: See NeuS2/LICENSE.txt
- 2DGS: See 2DGS/LICENSE.md
- PGSR: See PGSR/LICENSE.md
