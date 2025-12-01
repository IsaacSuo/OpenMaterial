# Quick Start Guide

## Overview

This repository now includes three additional state-of-the-art methods for benchmarking on the OpenMaterial dataset:

1. ✅ **Instant-NSR-PL** (baseline, already implemented)
2. ✅ **NeuS2** - Fast neural implicit surface reconstruction (ICCV 2023)
3. ✅ **2DGS** - 2D Gaussian Splatting (SIGGRAPH 2024)
4. ✅ **PGSR** - Planar-based Gaussian Splatting (TVCG 2024)

## Project Structure

```
OpenMaterial/
├── NeuS2/                          # NeuS2 repository (cloned)
├── 2DGS/                           # 2DGS repository (cloned)
├── PGSR/                           # PGSR repository (cloned)
├── run_neus2_openmaterial.sh       # Run NeuS2 on dataset
├── run_2dgs_openmaterial.sh        # Run 2DGS on dataset
├── run_pgsr_openmaterial.sh        # Run PGSR on dataset
├── evaluate_all_methods.sh         # Evaluate all methods
├── compare_methods.py              # Compare method results
├── METHODS_INTEGRATION.md          # Detailed integration guide
├── NeuS2_INTEGRATION.md            # NeuS2-specific documentation
└── NeuS2/tools/convert_openmaterial.py  # Data converter for NeuS2
```

## Installation

### 1. Clone the Repositories (Already Done)

The repositories have been cloned:
- `NeuS2/` - NeuS2 implementation
- `2DGS/` - 2D Gaussian Splatting
- `PGSR/` - PGSR implementation

### 2. Setup Environments on Server

**NeuS2:**
```bash
cd NeuS2
conda create -n neus2 python=3.9
conda activate neus2
pip install -r requirements.txt
pip install torch torchvision pytorch3d
cmake . -B build && cmake --build build --config RelWithDebInfo -j
cd ..
```

**2DGS:**
```bash
cd 2DGS
conda env create --file environment.yml
conda activate surfel_splatting
cd ..
```

**PGSR:**
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

## Usage

### Option 1: Run Single Method

```bash
# Run NeuS2 on objects 0-50 using GPU 0
bash run_neus2_openmaterial.sh 0 50 0

# Run 2DGS on objects 0-50 using GPU 0
bash run_2dgs_openmaterial.sh 0 50 0

# Run PGSR on objects 0-50 using GPU 0
bash run_pgsr_openmaterial.sh 0 50 0
```

### Option 2: Run Multiple Methods in Parallel

```bash
# Run all methods on different GPUs simultaneously
bash run_neus2_openmaterial.sh 0 50 0 &
bash run_2dgs_openmaterial.sh 0 50 1 &
bash run_pgsr_openmaterial.sh 0 50 2 &
wait
```

### Option 3: Run Baseline (Instant-NSR-PL)

```bash
cd instant-nsr-pl
bash run_openmaterial.sh 0 50 0
cd ..
```

## Evaluation

### Evaluate All Methods

```bash
bash evaluate_all_methods.sh
```

### Compare Methods

```bash
python compare_methods.py --methods instant-nsr-pl-wmask neus2 2dgs pgsr
```

This will generate:
- Console output with comparison tables
- `method_comparison.csv` with detailed results

## Expected Runtime

On a single RTX 3090 GPU for 50 objects:

| Method | Time per Scene | Total Time (50 objects × ~5 scenes) |
|--------|----------------|-------------------------------------|
| NeuS2 | ~5-10 min | ~20-40 hours |
| 2DGS | ~15-20 min | ~60-80 hours |
| PGSR | ~20-30 min | ~80-120 hours |

**Recommendation**: Use multiple GPUs to run methods in parallel!

## Output Files

After running, you will have:

```
Mesh/
├── neus2/meshes/          # NeuS2 reconstructed meshes
├── 2dgs/meshes/           # 2DGS reconstructed meshes
└── pgsr/meshes/           # PGSR reconstructed meshes

neus2_output/              # NeuS2 results and logs
2dgs_output/               # 2DGS results and logs
pgsr_output/               # PGSR results and logs

method_comparison.csv      # Comparison table
```

## Troubleshooting

### Build Errors

If you encounter build errors on the server:
- **NeuS2**: Requires CMake >= 3.18, CUDA >= 11.0
- **2DGS/PGSR**: Requires proper CUDA toolkit installation

```bash
# Check CUDA version
nvcc --version

# Check CMake version
cmake --version

# Upgrade CMake if needed
pip install cmake --upgrade
```

### Out of Memory

If you run out of GPU memory:
- Reduce batch size in method configs
- Process fewer objects at a time
- Use lower mesh resolution

### Data Issues

Make sure OpenMaterial dataset is downloaded:
```bash
python download.py --token YOUR_TOKEN --type all
```

## Next Steps

1. **Download Data**: Ensure OpenMaterial dataset is downloaded
2. **Setup Environments**: Install all dependencies on server
3. **Run Methods**: Execute batch scripts for each method
4. **Evaluate**: Run evaluation scripts
5. **Compare**: Analyze results using comparison script

## Documentation

- **Detailed Guide**: See `METHODS_INTEGRATION.md`
- **NeuS2 Details**: See `NeuS2_INTEGRATION.md`
- **Original README**: See `README.md`

## Support

If you encounter issues:
1. Check the detailed documentation in `METHODS_INTEGRATION.md`
2. Review method-specific README files in each repository
3. Check official repositories for known issues

## Key Scripts Summary

| Script | Purpose |
|--------|---------|
| `run_neus2_openmaterial.sh` | Train NeuS2 on OpenMaterial dataset |
| `run_2dgs_openmaterial.sh` | Train 2DGS on OpenMaterial dataset |
| `run_pgsr_openmaterial.sh` | Train PGSR on OpenMaterial dataset |
| `evaluate_all_methods.sh` | Evaluate geometry reconstruction |
| `compare_methods.py` | Generate comparison tables |
| `NeuS2/tools/convert_openmaterial.py` | Convert data for NeuS2 |

All scripts are executable and ready to use on the server!
