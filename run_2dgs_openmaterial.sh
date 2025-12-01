#!/bin/bash
# Run 2DGS on OpenMaterial dataset
# Usage: bash run_2dgs_openmaterial.sh START END GPU_ID

start=${1:-0}
end=${2:-10}
gpu=${3:-0}

# Paths
OPENMATERIAL_DIR="../datasets/openmaterial"
TWOGGS_DIR="./2DGS"
OUTPUT_DIR="./2DGS_output"
MESH_DIR="./Mesh/2dgs"

# Check if 2DGS environment is set up
if [ ! -d "$TWOGGS_DIR" ]; then
    echo "[ERROR] 2DGS directory not found!"
    echo "Please clone the repository first:"
    echo "  git clone --recursive https://github.com/hbb1/2d-gaussian-splatting.git 2DGS"
    exit 1
fi

# Check if OpenMaterial dataset exists
if [ ! -d "$OPENMATERIAL_DIR" ]; then
    echo "[ERROR] OpenMaterial dataset not found at $OPENMATERIAL_DIR"
    echo "Please download the dataset first using download.py"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$MESH_DIR/meshes"

# Get list of objects
objects=($(ls -d $OPENMATERIAL_DIR/*/ 2>/dev/null | xargs -n 1 basename))

if [ ${#objects[@]} -eq 0 ]; then
    echo "[ERROR] No objects found in $OPENMATERIAL_DIR"
    exit 1
fi

echo "[+] Found ${#objects[@]} objects in OpenMaterial dataset"
echo "[+] Processing objects from $start to $end on GPU $gpu"

count=0
for object_dir in "$OPENMATERIAL_DIR"/*; do
    object_name=$(basename "$object_dir")

    # Skip if out of range
    if [ $count -lt $start ]; then
        count=$((count+1))
        continue
    fi
    if [ $count -ge $end ]; then
        break
    fi

    echo ""
    echo "========================================"
    echo "[+] Processing object $count: $object_name"
    echo "========================================"

    # Process each scene for this object
    for scene_dir in "$object_dir"/*; do
        if [ ! -d "$scene_dir" ]; then
            continue
        fi

        scene_name=$(basename "$scene_dir")
        echo "[+] Scene: $scene_name"

        # Extract material name from scene name
        material_name=$(echo $scene_name | sed 's/.*-//')

        # 2DGS can directly use OpenMaterial data (COLMAP-compatible format)
        # Just need to make sure the structure is correct
        input_path="$scene_dir"
        model_output="$OUTPUT_DIR/$object_name/$scene_name"
        mesh_output="$MESH_DIR/meshes/$object_name"

        mkdir -p "$model_output"
        mkdir -p "$mesh_output"

        # Train 2DGS
        echo "  - Training 2DGS..."
        cd "$TWOGGS_DIR"

        CUDA_VISIBLE_DEVICES=$gpu python train.py \
            -s "../$input_path" \
            -m "../$model_output" \
            -r 1 \
            --iterations 30000 \
            --lambda_normal 0.05 \
            --lambda_distortion 1000 \
            --depth_ratio 0

        if [ $? -eq 0 ]; then
            echo "  [SUCCESS] Training completed"

            # Extract mesh
            echo "  - Extracting mesh..."
            CUDA_VISIBLE_DEVICES=$gpu python render.py \
                -s "../$input_path" \
                -m "../$model_output" \
                --skip_test \
                --skip_train \
                --mesh_res 1024 \
                --depth_ratio 0

            # Move mesh to output directory
            if [ -f "../$model_output/fuse.ply" ]; then
                cp "../$model_output/fuse.ply" "../$mesh_output/${material_name}.ply"
                echo "  [SUCCESS] Mesh extracted"
            else
                echo "  [WARNING] Mesh not found"
            fi

            # Save render results to output dir
            # Note: 2DGS doesn't directly output PSNR/SSIM, need to run evaluation
            mkdir -p "../$OUTPUT_DIR/$object_name"
            echo "$object_name:2dgs:$material_name:0.0-0.0" > "../$OUTPUT_DIR/$object_name/2dgs.txt"
        else
            echo "  [ERROR] Training failed"
        fi

        cd ..
    done

    count=$((count+1))
done

echo ""
echo "========================================"
echo "[+] Batch processing complete!"
echo "========================================"
echo "Models saved to: $OUTPUT_DIR"
echo "Meshes saved to: $MESH_DIR"
echo ""
echo "Next steps:"
echo "1. Evaluate meshes: bash Openmaterial-main/eval/eval.sh $MESH_DIR $OUTPUT_DIR 2dgs"
echo "2. Evaluate rendering quality: python 2DGS/metrics.py -m $OUTPUT_DIR"
echo "3. Summarize results: python sum_metrics.py --output_dir $OUTPUT_DIR"
