#!/bin/bash
# Unified evaluation script for all methods
# Usage: bash evaluate_all_methods.sh

echo "========================================="
echo " OpenMaterial Benchmark Evaluation"
echo "========================================="
echo ""

# Paths
MESH_BASE_DIR="./Mesh"
OUTPUT_BASE_DIR="./output"
EVAL_SCRIPT="./Openmaterial-main/eval/eval.sh"

# Methods to evaluate
methods=("neus2" "2dgs" "pgsr" "instant-nsr-pl-wmask")

echo "[+] Methods to evaluate: ${methods[@]}"
echo ""

# Check if evaluation script exists
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "[ERROR] Evaluation script not found at $EVAL_SCRIPT"
    exit 1
fi

# Evaluate each method
for method in "${methods[@]}"; do
    mesh_dir="$MESH_BASE_DIR/$method"
    output_dir="${method}_output"

    if [ ! -d "$mesh_dir" ]; then
        echo "[WARNING] Mesh directory not found for $method, skipping..."
        continue
    fi

    echo "========================================="
    echo " Evaluating $method"
    echo "========================================="

    # Run evaluation
    bash "$EVAL_SCRIPT" "$mesh_dir" "$output_dir" "$method"

    if [ $? -eq 0 ]; then
        echo "[SUCCESS] $method evaluation completed"
    else
        echo "[ERROR] $method evaluation failed"
    fi

    echo ""
done

echo "========================================="
echo " Generating Summary Reports"
echo "========================================="
echo ""

# Generate summaries
for method in "${methods[@]}"; do
    output_dir="${method}_output"

    if [ ! -d "$output_dir" ]; then
        continue
    fi

    echo "[+] Summarizing results for $method..."

    # For complete dataset
    if python sum_metrics.py --output_dir "$output_dir" 2>/dev/null; then
        echo "  [SUCCESS] Summary for complete dataset generated"
    fi

    # For ablation dataset (if applicable)
    if python sum_metrics-ablation.py --method "$method" --output_dir "$output_dir" --eval_mesh 2>/dev/null; then
        echo "  [SUCCESS] Summary for ablation dataset generated"
    fi

    echo ""
done

echo "========================================="
echo " Evaluation Complete!"
echo "========================================="
echo ""
echo "Results saved in:"
for method in "${methods[@]}"; do
    output_dir="${method}_output"
    if [ -d "$output_dir" ]; then
        echo "  - $output_dir/"
    fi
done
echo ""
echo "To compare methods, check the summary outputs above or run:"
echo "  python compare_methods.py"
