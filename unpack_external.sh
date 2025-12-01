#!/bin/bash
# 在服务器上解压 external 目录

echo "Unpacking external repositories..."

# 创建 external 目录
mkdir -p external

# 解压每个仓库
for tarfile in external_pack/*.tar.gz; do
    if [ -f "$tarfile" ]; then
        repo=$(basename "$tarfile" .tar.gz)
        echo "Unpacking $repo..."
        tar -xzf "$tarfile" -C external/
        echo "  ✓ Extracted to external/$repo"
    fi
done

echo ""
echo "All unpacked! Repositories in external/"
ls -1 external/
