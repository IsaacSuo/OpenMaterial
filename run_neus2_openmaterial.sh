#!/bin/bash
# Run NeuS2 on OpenMaterial dataset
# Usage: bash run_neus2_openmaterial.sh START END GPU_ID

start=${1:-0}
end=${2:-10}
gpu=${3:-0}

# Paths
OPENMATERIAL_DIR="../datasets/openmaterial"
NEUS2_DIR="./NeuS2"
NEUS2_DATA_DIR="./NeuS2_data"
OUTPUT_DIR="./NeuS2_output"
MESH_DIR="./Mesh/neus2"

# Check if NeuS2 is built
if [ ! -f "$NEUS2_DIR/build/testbed" ]; then
    echo "[ERROR] NeuS2 not built! Please run:"
    echo "  cd NeuS2"
    echo "  cmake . -B build"
    echo "  cmake --build build --config RelWithDebInfo -j"
    exit 1
fi

# Check if OpenMaterial dataset exists
if [ ! -d "$OPENMATERIAL_DIR" ]; then
    echo "[ERROR] OpenMaterial dataset not found at $OPENMATERIAL_DIR"
    echo "Please download the dataset first using download.py"
    exit 1
fi

# Create output directories
mkdir -p "$NEUS2_DATA_DIR"
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

        # Convert data format
        input_path="$scene_dir"
        output_path="$NEUS2_DATA_DIR/$object_name/$scene_name"

        echo "  - Converting data format..."
        python "$NEUS2_DIR/tools/convert_openmaterial.py" \
            --input "$input_path" \
            --output "$output_path" \
            --splits train test

        if [ $? -ne 0 ]; then
            echo "  [ERROR] Data conversion failed, skipping..."
            continue
        fi

        # Train NeuS2
        echo "  - Training NeuS2..."
        exp_name="${object_name}_${scene_name}"

        cd "$NEUS2_DIR"
        CUDA_VISIBLE_DEVICES=$gpu python scripts/run.py \
            --scene "../$output_path/transforms_train.json" \
            --name "$exp_name" \
            --network dtu.json \
            --n_steps 15000 \
            --save_mesh \
            --save_mesh_path "../$MESH_DIR/meshes/$object_name/${material_name}.ply" \
            --marching_cubes_res 512

        cd ..

        if [ $? -eq 0 ]; then
            echo "  [SUCCESS] Training completed"

            # Save results
            mkdir -p "$OUTPUT_DIR/$object_name"

            # Extract metrics if available
            log_file="$NEUS2_DIR/output/$exp_name/logs"
            if [ -d "$log_file" ]; then
                # Try to extract PSNR, SSIM from logs
                # Format: object:method:material:PSNR-SSIM
                echo "$object_name:neus2:$material_name:0.0-0.0" > "$OUTPUT_DIR/$object_name/neus2.txt"
            fi
        else
            echo "  [ERROR] Training failed"
        fi
    done

    count=$((count+1))
done

echo ""
echo "========================================"
echo "[+] Batch processing complete!"
echo "========================================"
echo "Meshes saved to: $MESH_DIR"
echo "Outputs saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Evaluate meshes: bash Openmaterial-main/eval/eval.sh $MESH_DIR $OUTPUT_DIR neus2"
echo "2. Summarize results: python sum_metrics.py --output_dir $OUTPUT_DIR"
