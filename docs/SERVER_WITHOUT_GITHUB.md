# 服务器无法访问 GitHub 的部署方案

如果你的服务器无法访问 GitHub（常见于内网服务器或有网络限制的环境），可以使用打包上传的方式。

## 方案概述

1. **本地**：打包 `external/` 目录（排除 `.git` 等大文件）
2. **Git**：将打包文件提交到仓库
3. **服务器**：从打包文件解压

## 详细步骤

### 步骤 1：本地打包（一次性操作）

```bash
# 在本地项目根目录执行
./pack_external.sh
```

这会创建 `external_pack/` 目录，包含：
- `NeuS2.tar.gz` (~154MB)
- `2DGS.tar.gz` (~14MB)
- `PGSR.tar.gz` (~5MB)

**总大小**: ~172MB（已排除 `.git` 目录，原始大小为 741MB）

### 步骤 2：提交并推送

```bash
git add external_pack/
git add pack_external.sh unpack_external.sh
git commit -m "Add packed external repositories for servers without GitHub access"
git push
```

### 步骤 3：服务器部署

```bash
# 1. 克隆/更新代码
git clone YOUR_REPO OpenMaterial
# 或者如果已有：
cd OpenMaterial && git pull

# 2. 解压外部仓库
python setup_methods.py --unpack

# 3. 下载数据集
python download.py --token YOUR_HF_TOKEN --type ablation

# 4. 设置环境
python setup_methods.py --setup all

# 5. 运行基准测试
python run_benchmark.py --method all --start 0 --end 50 --gpus 0,1,2
```

## 方案对比

### 方案 A：直接 Clone（需要 GitHub 访问）

```bash
python setup_methods.py --clone
python setup_methods.py --setup all
```

**优点**：
- ✅ 最简单
- ✅ 总是获取最新版本

**缺点**：
- ❌ 需要 GitHub 访问
- ❌ 下载慢（如果网络不好）

### 方案 B：打包上传（无需 GitHub 访问）✅

```bash
# 本地
./pack_external.sh
git add external_pack/
git push

# 服务器
python setup_methods.py --unpack
python setup_methods.py --setup all
```

**优点**：
- ✅ 不需要 GitHub 访问
- ✅ 通过 Git 传输，利用已有通道
- ✅ 大小适中（172MB vs 741MB）

**缺点**：
- ⚠️ 需要一次性设置（打包）
- ⚠️ 更新外部仓库需要重新打包

## 技术细节

### pack_external.sh 做了什么？

```bash
tar --exclude='.git' \          # 排除 Git 历史（305MB）
    --exclude='__pycache__' \   # 排除 Python 缓存
    --exclude='*.pyc' \         # 排除编译的 Python 文件
    --exclude='*.so' \          # 排除编译的库
    --exclude='*.o' \           # 排除目标文件
    --exclude='build' \         # 排除构建目录
    --exclude='*.egg-info' \    # 排除包信息
    -czf "external_pack/${repo}.tar.gz" \
    -C external "$repo"
```

**结果**：
- 原始大小：741MB
- 打包后大小：172MB
- 节省：77%

### 为什么不用 Git LFS？

Git LFS（Large File Storage）也是一个选择，但：
- ❌ 需要额外配置
- ❌ 有些 Git 服务器不支持
- ✅ tar.gz 更通用

### 为什么不直接提交 external/？

| 项目 | 提交 external/ | 使用 external_pack/ |
|------|----------------|-------------------|
| Git 仓库大小 | +1GB | +172MB |
| Clone 时间 | 很慢 | 适中 |
| .git 嵌套问题 | ✅ 有 | ❌ 无 |
| 编译产物问题 | ✅ 有 | ❌ 无 |

## 更新外部仓库

如果需要更新外部仓库（如 NeuS2 有新版本）：

### 本地

```bash
# 1. 删除旧的打包
rm -rf external_pack/

# 2. 更新仓库
cd external/NeuS2
git pull
cd ../..

# 3. 重新打包
./pack_external.sh

# 4. 提交
git add external_pack/
git commit -m "Update NeuS2 to latest version"
git push
```

### 服务器

```bash
git pull
python setup_methods.py --clean  # 删除旧的 external/
python setup_methods.py --unpack # 解压新的
python setup_methods.py --setup all  # 重新设置环境
```

## 故障排除

### 问题 1：解压失败

```bash
# 检查打包文件是否存在
ls external_pack/

# 应该看到：
# 2DGS.tar.gz
# NeuS2.tar.gz
# PGSR.tar.gz
```

### 问题 2：打包文件损坏

```bash
# 验证打包文件完整性
tar -tzf external_pack/NeuS2.tar.gz | head -10

# 如果报错，重新打包
rm external_pack/NeuS2.tar.gz
./pack_external.sh
```

### 问题 3：Git 仓库过大

如果你的 Git 仓库变得很大：

```bash
# 检查仓库大小
du -sh .git

# 如果需要，清理历史
git gc --aggressive --prune=now
```

## 混合方案

如果部分服务器能访问 GitHub，部分不能：

```bash
# 能访问 GitHub 的服务器
python setup_methods.py --clone

# 不能访问 GitHub 的服务器
python setup_methods.py --unpack
```

两种方式设置的环境完全一样，可以自由选择。

## 总结

对于**无法访问 GitHub 的服务器**，推荐使用打包上传方案：

1. ✅ 本地打包：`./pack_external.sh`
2. ✅ Git 提交：`git add external_pack/ && git push`
3. ✅ 服务器解压：`python setup_methods.py --unpack`
4. ✅ 设置环境：`python setup_methods.py --setup all`

这是目前最简单、最可靠的方案。
