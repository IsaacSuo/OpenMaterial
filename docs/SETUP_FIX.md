# Setup方法依赖修复说明

## 问题描述

之前各个方法的 `setup()` 函数存在缺少必要依赖库的问题，导致环境配置不完整，运行时会出现 `ModuleNotFoundError` 等错误。

## 修复内容

### 1. NeuS2 方法 (`methods/wrappers/neus2_wrapper.py`)

**新增依赖:**
- `trimesh` - 用于mesh文件的加载和处理
- `tensorboard` - 用于训练过程可视化

**修复位置:** neus2_wrapper.py:37

### 2. 2DGS 方法 (`methods/wrappers/twodgs_wrapper.py`)

**新增依赖:**
- `open3d==0.18.0` - 用于3D点云和mesh处理（render.py需要）
- `lpips` - 用于感知损失计算
- `scikit-image` - 用于图像处理
- `trimesh` - 用于mesh处理

**修复位置:** twodgs_wrapper.py:48-54

### 3. PGSR 方法 (`methods/wrappers/pgsr_wrapper.py`)

**新增依赖:**
- `pytorch3d` - 用于3D深度学习操作（mesh处理需要）
  - 优先从PyPI镜像安装
  - 失败则从GitHub源码安装

**修复位置:** pgsr_wrapper.py:56-68

### 4. Instant-NSR-PL 方法 (`methods/wrappers/instant_nsr_wrapper.py`)

**完全重写setup()方法:**

之前的实现只是打印了一条消息就返回True，实际上没有安装任何依赖。

**新增功能:**
1. 创建 conda 环境 (如果不存在)
2. 安装 PyTorch 2.3.1 (检查已安装版本，避免重复下载)
3. 从 requirements.txt 安装所有依赖

**修复位置:** instant_nsr_wrapper.py:20-59

## 验证方法

### 使用检查脚本

```bash
# 检查所有环境
python check_dependencies.py

# 检查特定环境
python check_dependencies.py --env neus2
python check_dependencies.py --env surfel_splatting  # 2DGS
python check_dependencies.py --env pgsr
python check_dependencies.py --env instant-nsr-pl
```

### 重新运行setup

```bash
# 重新设置所有方法
python setup_methods.py --setup all

# 或单独设置某个方法
python setup_methods.py --setup neus2
python setup_methods.py --setup 2dgs
python setup_methods.py --setup pgsr
python setup_methods.py --setup instant-nsr-pl
```

## 环境配置对照表

| 方法 | Conda环境名 | Python | PyTorch | 关键依赖 |
|------|------------|--------|---------|---------|
| NeuS2 | neus2 | 3.9 | 2.3.1 | pytorch3d, trimesh, tensorboard |
| 2DGS | surfel_splatting | 3.8 | 2.3.1 | open3d, lpips, trimesh |
| PGSR | pgsr | 3.8 | 2.3.1 | pytorch3d, open3d, lpips, trimesh |
| Instant-NSR-PL | instant-nsr-pl | 3.8 | 2.3.1 | nerfacc, pytorch_lightning |

## 训练日志优化

**问题:** 训练时进度条大量输出占满控制台

**解决方案:** 修改了 `methods/base_method.py` 的日志输出方式：
- ✅ 进度信息只在单行动态更新（使用 `\r` 覆盖）
- ✅ 自动识别进度模式：Iteration X/Y, Step X, Epoch X等
- ✅ 重要信息（error, warning）仍会换行显示
- ✅ 完整日志保存到文件，方便调试

**效果:**
```bash
Training logs: benchmark_output/2dgs/models/obj/scene/training.log
Iteration 15230/30000 Loss: 0.0234 PSNR: 28.45 SSIM: 0.923  # 动态更新
Training completed with return code: 0
```

## 断点续训功能

**新增功能:** 支持跳过已完成的场景，实现断点续训

**使用方法:**
```bash
# 跳过已经有mesh输出的场景
python run_benchmark.py \
    --method 2dgs \
    --dataset datasets/openmaterial \
    --skip-completed \
    --gpu 0

# 组合使用：跳过已完成的场景 + 跳过已训练的模型
python run_benchmark.py \
    --method 2dgs \
    --dataset datasets/openmaterial \
    --skip-completed \
    --skip-train \
    --gpu 0
```

**特性:**
- ✅ 每个场景处理完立即保存到 `benchmark_results.json`
- ✅ 中断后重新运行，使用 `--skip-completed` 自动跳过已完成的场景
- ✅ 判断依据：mesh文件是否存在（`benchmark_output/{method}/meshes/{object}/{scene}.ply`）
- ✅ 跳过的场景也会记录到结果文件中，标记为 `skipped: true`

**对比:**

| 参数 | 作用 | 检查内容 |
|------|------|---------|
| `--skip-train` | 跳过训练步骤 | 检查checkpoint文件 |
| `--skip-mesh` | 跳过mesh提取 | 不提取mesh |
| `--skip-completed` | 跳过整个场景 | 检查最终mesh文件 |
