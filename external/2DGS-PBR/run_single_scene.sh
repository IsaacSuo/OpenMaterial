#!/bin/bash
# 单场景 2DGS-PBR 测试脚本
# Usage: bash run_single_scene.sh <数据集路径> <输出路径> [迭代次数]

DATASET=$1
OUTPUT=$2
ITERATIONS=${3:-30000}

if [ -z "$DATASET" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: bash run_single_scene.sh <dataset_path> <output_path> [iterations]"
    echo "Example: bash run_single_scene.sh /path/to/scene /path/to/output 30000"
    exit 1
fi

echo "=== 2DGS-PBR Single Scene Test ==="
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
echo "Iterations: $ITERATIONS"

# 训练
echo ""
echo "=== Training ==="
python train_pbr.py \
    -s "$DATASET" \
    -m "$OUTPUT" \
    --iterations $ITERATIONS \
    --test_iterations 7000 $ITERATIONS \
    --save_iterations 7000 $ITERATIONS

# 渲染
echo ""
echo "=== Rendering ==="
python render_pbr.py \
    -m "$OUTPUT" \
    --compute_metrics

echo ""
echo "=== Done ==="
echo "Results saved to: $OUTPUT"
