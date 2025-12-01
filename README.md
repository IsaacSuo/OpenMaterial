# OpenMaterial Benchmark - 模块化版本

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-orange.svg)](https://developer.nvidia.com/cuda-toolkit)

本项目为 OpenMaterial 数据集提供了统一的基准测试框架，支持多种3D重建方法。

## ✨ 特性

- 🎯 **统一接口**: 所有方法通过统一 API 调用
- 🔌 **模块化架构**: 外部仓库独立管理，无 Git 冲突
- 🚀 **易于扩展**: 添加新方法只需实现 wrapper
- ⚡ **并行运行**: 支持多 GPU 并行训练
- 📊 **自动评估**: 内置评估和结果对比
- 🛠️ **灵活配置**: 支持 JSON 配置文件

## 📦 支持的方法

| 方法 | 类型 | 训练时间 | 特点 |
|------|------|---------|------|
| **NeuS2** | Neural Implicit Surface | ~5-10 min | 最快速度 |
| **2DGS** | 2D Gaussian Splatting | ~15-20 min | 最佳几何质量 |
| **PGSR** | Planar Gaussian Splatting | ~20-30 min | 适合平面场景 |
| **Instant-NSR-PL** | Neural Implicit Surface | ~5-10 min | 基线方法 |

## 🚀 快速开始

### 安装

```bash
# 1. 克隆项目（包含所有外部仓库）
git clone YOUR_REPO_URL
cd OpenMaterial

# 2. 下载数据
python download.py --token YOUR_HF_TOKEN --type ablation

# 3. 设置环境
python setup_methods.py --setup all
```

### 运行

```bash
# 单个方法
python run_benchmark.py --method neus2 --start 0 --end 50 --gpu 0

# 所有方法（并行）
python run_benchmark.py --method all --start 0 --end 50 --gpus 0,1,2

# 评估结果
bash evaluate_all_methods.sh
python compare_methods.py
```

## 📁 项目结构

```
OpenMaterial/
├── methods/                    # 方法接口层
│   ├── base_method.py         # 基类
│   └── wrappers/              # 方法包装器
│       ├── neus2_wrapper.py
│       ├── twodgs_wrapper.py
│       ├── pgsr_wrapper.py
│       └── instant_nsr_wrapper.py
├── external/                   # 外部仓库（自动克隆）
│   ├── NeuS2/
│   ├── 2DGS/
│   └── PGSR/
├── setup_methods.py           # 仓库管理脚本
├── run_benchmark.py           # 统一运行脚本
├── evaluate_all_methods.sh    # 评估脚本
├── compare_methods.py         # 结果对比
└── datasets/                  # 数据集目录
    └── openmaterial/
```

## 📖 文档

- 🚀 **[快速开始](docs/QUICKSTART.md)** - 30秒上手
- 📘 **[详细部署](docs/DEPLOYMENT.md)** - 完整部署指南

## 🎯 使用示例

### 命令行

```bash
# 运行 NeuS2
python run_benchmark.py \
    --method neus2 \
    --dataset datasets/openmaterial \
    --output results \
    --start 0 --end 50 \
    --gpu 0

# 并行运行所有方法
python run_benchmark.py \
    --method all \
    --start 0 --end 50 \
    --gpus 0,1,2

# 使用配置文件
python run_benchmark.py \
    --method neus2 \
    --config configs/fast.json \
    --gpu 0
```

### Python API

```python
from methods import get_method

# 初始化方法
neus2 = get_method('neus2')(repo_path='external/NeuS2')

# 设置环境（首次运行）
neus2.setup()

# 处理场景
result = neus2.process_scene(
    input_scene='datasets/openmaterial/obj/scene',
    output_dir='results',
    gpu_id=0,
    n_steps=15000
)

print(f"Mesh: {result['mesh_output']}")
```

## 🔧 配置

创建 JSON 配置文件：

```json
{
    "n_steps": 20000,
    "marching_cubes_res": 1024,
    "learning_rate": 0.01
}
```

使用：
```bash
python run_benchmark.py --method neus2 --config my_config.json --gpu 0
```

## 📊 评估

```bash
# 评估所有方法
bash evaluate_all_methods.sh

# 对比结果
python compare_methods.py --methods instant-nsr-pl neus2 2dgs pgsr

# 输出
# ===============================================
#  PSNR (Peak Signal-to-Noise Ratio) ↑
# ===============================================
#                 diffuse  conductor  dielectric  ...
# instant-nsr-pl    30.2      28.5       27.3    ...
# neus2             30.5      28.8       27.6    ...
# 2dgs              32.1      30.2       29.1    ...
# pgsr              31.8      29.9       28.8    ...
```

## 🌟 优势

### vs 嵌入式架构

| 特性 | 嵌入式 | 模块化 |
|------|--------|--------|
| Git 管理 | 复杂（子模块） | ✅ 简单 |
| 接口统一 | ❌ 无 | ✅ 有 |
| 扩展性 | ❌ 困难 | ✅ 容易 |
| 错误处理 | ❌ 基础 | ✅ 完善 |
| 并行运行 | ⚠️ 手动 | ✅ 自动 |

### 关键改进

- ✅ **无 Git 冲突**: 外部仓库独立管理
- ✅ **统一接口**: 所有方法相同 API
- ✅ **自动化**: 环境设置、训练、评估全自动
- ✅ **模块化**: 易于添加新方法
- ✅ **灵活性**: 支持配置文件和 Python API

## 🚢 服务器部署

```bash
# 服务器操作
ssh user@server
git clone YOUR_REPO_URL
cd OpenMaterial

# 下载数据
python download.py --token YOUR_HF_TOKEN --type all

# 设置环境
python setup_methods.py --setup all

# 使用 tmux 运行
tmux new -s benchmark
python run_benchmark.py --method all --start 0 --end 100 --gpus 0,1,2
# Ctrl+B, D

# 监控
tail -f benchmark_output/*/benchmark_results.json
watch -n 1 nvidia-smi
```

## 🛠️ 添加新方法

1. 创建 wrapper：

```python
# methods/wrappers/mymethod_wrapper.py
from ..base_method import BaseMethod

class MyMethod(BaseMethod):
    def setup(self) -> bool:
        # 设置环境
        pass

    def convert_data(self, input_path, output_path) -> bool:
        # 转换数据
        pass

    def train(self, data_path, output_path, **kwargs) -> bool:
        # 训练
        pass

    def extract_mesh(self, model_path, output_mesh_path, **kwargs) -> bool:
        # 提取 mesh
        pass

    def get_default_config(self):
        return {'param': value}
```

2. 注册方法：

```python
# methods/__init__.py
from .wrappers.mymethod_wrapper import MyMethod

METHODS = {
    ...
    'mymethod': MyMethod,
}
```

3. 使用：

```bash
python run_benchmark.py --method mymethod --gpu 0
```

## 📈 性能对比

在 RTX 3090 上处理 50 个物体（~250 场景）：

| 配置 | 嵌入式 | 模块化 |
|------|--------|--------|
| 单 GPU 顺序 | ~30天 | ~30天 |
| 3 GPU 并行 | ~10天（手动） | ~10天（自动） |
| 易用性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 可维护性 | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🐛 故障排除

### 环境问题

```bash
# 检查环境
conda env list

# 重新设置
python setup_methods.py --setup neus2

# 测试
conda activate neus2
python -c "import torch; print(torch.cuda.is_available())"
```

### 仓库问题

```bash
# 重新克隆
python setup_methods.py --clean
python setup_methods.py --clone

# 检查
ls external/
```

### 运行问题

```bash
# 查看详细错误
python run_benchmark.py --method neus2 --start 0 --end 1 --gpu 0

# 干运行测试
python run_benchmark.py --method neus2 --dry-run
```

## 📜 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE)。

外部方法保留其原始许可证：
- NeuS2: [License](external/NeuS2/LICENSE.txt)
- 2DGS: [License](external/2DGS/LICENSE.md)
- PGSR: [License](external/PGSR/LICENSE.md)

## 🙏 致谢

- [NeuS2](https://github.com/19reborn/NeuS2) - Fast neural surface reconstruction
- [2DGS](https://github.com/hbb1/2d-gaussian-splatting) - 2D Gaussian Splatting
- [PGSR](https://github.com/zju3dv/PGSR) - Planar-based Gaussian Splatting
- [OpenMaterial](https://openmaterial.github.io/) - Dataset

## 📮 联系

如有问题或建议，请提交 Issue 或 Pull Request。

---

**从这里开始**: [快速开始指南](docs/QUICKSTART.md) | [详细文档](docs/DEPLOYMENT.md)
