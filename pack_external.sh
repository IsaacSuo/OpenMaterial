#!/bin/bash
# 打包 external 目录，排除不必要的文件

echo "Packing external repositories..."

# 创建临时目录
mkdir -p external_pack

# 打包每个仓库（排除 .git 和编译产物）
for repo in NeuS2 2DGS PGSR; do
    if [ -d "external/$repo" ]; then
        echo "Packing $repo..."
        tar --exclude='.git' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='*.so' \
            --exclude='*.o' \
            --exclude='build' \
            --exclude='*.egg-info' \
            -czf "external_pack/${repo}.tar.gz" \
            -C external "$repo"
        echo "  ✓ Created external_pack/${repo}.tar.gz"
    fi
done

echo ""
echo "All packed! Files in external_pack/"
ls -lh external_pack/
