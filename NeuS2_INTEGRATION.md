# NeuS2 Integration for OpenMaterial Dataset

This document explains how to use NeuS2 for benchmarking on the OpenMaterial dataset.

## Setup

### 1. Build NeuS2

```bash
cd NeuS2

# Install dependencies
conda create -n neus2 python=3.9
conda activate neus2
pip install -r requirements.txt
pip install torch torchvision pytorch3d

# Build CUDA code (requires CMake >= 3.18, CUDA >= 11.0)
cmake . -B build
cmake --build build --config RelWithDebInfo -j
```

### 2. Verify Installation

```bash
# Test if NeuS2 is built successfully
./build/testbed --help
```

## Running NeuS2 on OpenMaterial

### Single Scene

1. **Convert data format:**
```bash
python NeuS2/tools/convert_openmaterial.py \
    --input datasets/openmaterial/OBJECT_ID/SCENE_NAME \
    --output NeuS2_data/OBJECT_ID/SCENE_NAME \
    --splits train test
```

2. **Train:**
```bash
cd NeuS2
python scripts/run.py \
    --scene ../NeuS2_data/OBJECT_ID/SCENE_NAME/transforms_train.json \
    --name experiment_name \
    --network dtu.json \
    --n_steps 15000 \
    --save_mesh \
    --save_mesh_path ../meshes/object_scene.ply \
    --marching_cubes_res 512
```

### Batch Processing (Complete Dataset)

```bash
# Process objects 0-50 on GPU 0
bash run_neus2_openmaterial.sh 0 50 0

# Process objects 50-100 on GPU 1
bash run_neus2_openmaterial.sh 50 100 1
```

### Ablation Dataset

```bash
# For ablation dataset
bash run_neus2_openmaterial.sh 0 5 0
```

## Data Format

### OpenMaterial Format
```
datasets/openmaterial/OBJECT_ID/SCENE_NAME/
├── train/
│   ├── images/
│   └── mask/
├── test/
│   ├── images/
│   └── mask/
├── transforms_train.json
└── transforms_test.json
```

### NeuS2 Format (after conversion)
```
NeuS2_data/OBJECT_ID/SCENE_NAME/
├── images/
│   ├── 000000.png  (RGBA: RGB + mask as alpha channel)
│   ├── 000001.png
│   └── ...
├── transforms_train.json
└── transforms_test.json
```

## Output Structure

```
NeuS2_output/
├── OBJECT_ID/
│   └── neus2.txt  (format: object:method:material:PSNR-SSIM)

Mesh/neus2/
├── meshes/
│   ├── OBJECT_ID/
│   │   └── material_name.ply

NeuS2/output/
├── experiment_name/
│   ├── checkpoints/
│   ├── mesh/
│   └── logs/
```

## Evaluation

### 1. Evaluate Geometry (Chamfer Distance)

```bash
bash Openmaterial-main/eval/eval.sh ./Mesh/neus2 ./NeuS2_output neus2
```

### 2. Summarize Results

```bash
# For complete dataset
python sum_metrics.py --output_dir NeuS2_output

# For ablation dataset
python sum_metrics-ablation.py --method neus2 --output_dir NeuS2_output --eval_mesh
```

## Configuration

### Training Parameters

Edit `NeuS2/configs/nerf/dtu.json` to adjust:
- `n_steps`: Training iterations (default: 15000)
- `learning_rate`: Learning rate
- `batch_size`: Rays per batch
- `marching_cubes_res`: Mesh resolution (default: 512)

### For Different Materials

NeuS2 uses the same configuration for all materials. Performance may vary:
- **Best**: Diffuse, Rough Plastic
- **Challenging**: Conductor, Dielectric (high specular reflection)

## Troubleshooting

### Build Issues

1. **CMake version too old:**
```bash
# Install newer CMake
pip install cmake --upgrade
```

2. **CUDA not found:**
```bash
# Set CUDA path
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

3. **Submodule errors:**
```bash
cd NeuS2
git submodule update --init --recursive
```

### Runtime Issues

1. **Out of memory:**
   - Reduce `marching_cubes_res` (e.g., 256 instead of 512)
   - Reduce batch size in config

2. **Poor reconstruction:**
   - Increase `n_steps` (e.g., 20000-30000)
   - Check if masks are correctly loaded
   - Adjust `aabb_scale` in transforms.json

## Performance Expectations

- **Training Time**: ~5-10 minutes per scene (RTX 3090)
- **Memory Usage**: ~8-12GB GPU memory
- **Mesh Quality**: Comparable to NeuS but 100x faster

## Notes

- NeuS2 requires masks for good results
- Use `--white_bkgd` flag for transparent objects
- For dynamic scenes, see `scripts/run_dynamic.py`
- Pretrained models available at project website
