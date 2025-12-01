## 模块化部署指南

本项目已重构为模块化架构，外部方法仓库作为独立模块管理，避免 Git 子模块冲突。

## 🏗️ 新架构

```
OpenMaterial/
├── methods/                    # 统一方法接口层
│   ├── __init__.py            # 方法注册
│   ├── base_method.py         # 基类
│   └── wrappers/              # 各方法包装器
│       ├── neus2_wrapper.py
│       ├── twodgs_wrapper.py
│       ├── pgsr_wrapper.py
│       └── instant_nsr_wrapper.py
├── external/                   # 外部仓库（独立管理）
│   ├── NeuS2/                 # NeuS2 仓库
│   ├── 2DGS/                  # 2DGS 仓库
│   └── PGSR/                  # PGSR 仓库
├── setup_methods.py           # 仓库管理脚本
├── run_benchmark.py           # 统一运行脚本
└── instant-nsr-pl/            # 基线方法（已有）
```

## 🚀 快速开始

### 1. 设置外部仓库

#### 方案 A: 如果您已经有嵌入的仓库（从之前 git clone 的）

```bash
# 移动到 external/ 目录
python setup_methods.py --move

# 清理旧目录
rm -rf NeuS2/ 2DGS/ PGSR/

# 确认结构
ls external/
# 应该看到: NeuS2  2DGS  PGSR
```

#### 方案 B: 全新克隆

```bash
# 克隆所有外部仓库到 external/
python setup_methods.py --clone

# 这会自动：
# - 创建 external/ 目录
# - 克隆 NeuS2, 2DGS, PGSR
# - 初始化子模块
```

### 2. 设置方法环境

```bash
# 设置所有方法
python setup_methods.py --setup all

# 或设置特定方法
python setup_methods.py --setup neus2
python setup_methods.py --setup 2dgs pgsr
```

这会自动：
- 创建 conda 环境
- 安装依赖
- 编译 CUDA 代码

### 3. 运行基准测试

```bash
# 运行单个方法
python run_benchmark.py --method neus2 --start 0 --end 50 --gpu 0

# 运行所有方法（并行）
python run_benchmark.py --method all --start 0 --end 50 --gpus 0,1,2

# 带配置文件运行
python run_benchmark.py --method neus2 --config configs/neus2_custom.json --gpu 0
```

## 📝 详细命令说明

### setup_methods.py

```bash
# 克隆外部仓库
python setup_methods.py --clone

# 移动嵌入的仓库到 external/
python setup_methods.py --move

# 设置环境
python setup_methods.py --setup neus2
python setup_methods.py --setup all

# 清理（删除 external/）
python setup_methods.py --clean
```

### run_benchmark.py

```bash
# 基本用法
python run_benchmark.py \
    --method neus2 \
    --dataset datasets/openmaterial \
    --output benchmark_output \
    --start 0 \
    --end 50 \
    --gpu 0

# 运行所有方法（并行）
python run_benchmark.py \
    --method all \
    --start 0 \
    --end 50 \
    --gpus 0,1,2,3

# 只训练，不提取 mesh
python run_benchmark.py \
    --method neus2 \
    --only-train \
    --gpu 0

# 跳过环境设置（如果已设置）
python run_benchmark.py \
    --method neus2 \
    --skip-setup \
    --gpu 0

# 干运行（查看会处理什么）
python run_benchmark.py \
    --method neus2 \
    --dry-run
```

## 🔧 方法配置

创建 JSON 配置文件来自定义参数：

```json
// configs/neus2_fast.json
{
    "n_steps": 10000,
    "marching_cubes_res": 256
}

// configs/2dgs_highquality.json
{
    "iterations": 40000,
    "lambda_normal": 0.1,
    "mesh_res": 2048
}
```

使用配置：
```bash
python run_benchmark.py \
    --method neus2 \
    --config configs/neus2_fast.json \
    --gpu 0
```

## 🎯 模块化 API 使用示例

您也可以在 Python 代码中直接使用方法：

```python
from methods import get_method

# 获取方法类
NeuS2 = get_method('neus2')

# 初始化
method = NeuS2(repo_path='external/NeuS2')

# 设置环境（首次运行）
method.setup()

# 处理单个场景
result = method.process_scene(
    input_scene='datasets/openmaterial/obj_001/scene_001',
    output_dir='my_output',
    gpu_id=0,
    n_steps=15000
)

print(result)
# {
#     'scene': 'scene_001',
#     'success': True,
#     'mesh_output': 'my_output/meshes/obj_001/scene_001.ply',
#     ...
# }
```

更底层的控制：

```python
from methods.wrappers.neus2_wrapper import NeuS2Method

method = NeuS2Method(repo_path='external/NeuS2')

# 1. 转换数据
method.convert_data(
    input_path='datasets/openmaterial/obj_001/scene_001',
    output_path='converted_data/scene_001'
)

# 2. 训练
method.train(
    data_path='converted_data/scene_001',
    output_path='models/scene_001',
    n_steps=20000
)

# 3. 提取 mesh
method.extract_mesh(
    model_path='models/scene_001',
    output_mesh_path='meshes/scene_001.ply',
    marching_cubes_res=1024
)
```

## 📊 输出结构

```
benchmark_output/
├── neus2/
│   ├── converted_data/       # 转换后的数据
│   ├── models/                # 训练的模型
│   ├── meshes/                # 导出的 mesh
│   └── benchmark_results.json # 结果摘要
├── 2dgs/
│   ├── models/
│   ├── meshes/
│   └── benchmark_results.json
└── pgsr/
    ├── models/
    ├── meshes/
    └── benchmark_results.json
```

## 🔄 从旧脚本迁移

### 旧方式（嵌入式）

```bash
# 旧的运行脚本
bash run_neus2_openmaterial.sh 0 50 0
bash run_2dgs_openmaterial.sh 0 50 1
bash run_pgsr_openmaterial.sh 0 50 2
```

### 新方式（模块化）

```bash
# 新的统一脚本
python run_benchmark.py --method all --start 0 --end 50 --gpus 0,1,2
```

优势：
- ✅ 统一接口
- ✅ 更好的错误处理
- ✅ 自动保存结果
- ✅ 进度条显示
- ✅ 并行运行支持
- ✅ 配置文件支持

## 🛠️ 故障排除

### 问题：找不到方法模块

```bash
# 确保在项目根目录
cd /path/to/OpenMaterial

# 检查 methods/ 目录
ls methods/wrappers/

# 应该看到:
# __init__.py
# neus2_wrapper.py
# twodgs_wrapper.py
# pgsr_wrapper.py
# instant_nsr_wrapper.py
```

### 问题：external/ 目录为空

```bash
# 克隆仓库
python setup_methods.py --clone

# 或移动现有仓库
python setup_methods.py --move
```

### 问题：方法设置失败

```bash
# 单独设置每个方法，查看详细错误
python setup_methods.py --setup neus2

# 检查 conda 环境
conda env list

# 手动激活环境测试
conda activate neus2
python -c "import torch; print(torch.cuda.is_available())"
```

### 问题：运行时找不到仓库

检查 `methods/wrappers/*_wrapper.py` 中的 `repo_path` 参数是否正确：

```python
# 应该是相对于项目根目录的路径
NeuS2Method(repo_path='external/NeuS2')  # ✓ 正确
NeuS2Method(repo_path='NeuS2')           # ✗ 错误（如果已移动）
```

## 📋 完整部署流程

```bash
# 1. 上传到服务器
rsync -avz OpenMaterial/ user@server:/path/to/OpenMaterial/

# 2. 登录服务器
ssh user@server
cd /path/to/OpenMaterial

# 3. 下载数据
python download.py --token YOUR_TOKEN --type ablation

# 4. 设置外部仓库
python setup_methods.py --clone

# 5. 设置环境
python setup_methods.py --setup all

# 6. 测试单个场景
python run_benchmark.py \
    --method neus2 \
    --start 0 \
    --end 1 \
    --gpu 0 \
    --dry-run

# 7. 运行完整基准测试
python run_benchmark.py \
    --method all \
    --start 0 \
    --end 50 \
    --gpus 0,1,2

# 8. 评估结果
bash evaluate_all_methods.sh
python compare_methods.py
```

## 🎓 旧脚本兼容性

旧的 shell 脚本仍然可用，但需要调整路径：

```bash
# 编辑旧脚本，更新路径
sed -i 's|NEUS2_DIR="./NeuS2"|NEUS2_DIR="./external/NeuS2"|g' run_neus2_openmaterial.sh
sed -i 's|TWOGGS_DIR="./2DGS"|TWOGGS_DIR="./external/2DGS"|g' run_2dgs_openmaterial.sh
sed -i 's|PGSR_DIR="./PGSR"|PGSR_DIR="./external/PGSR"|g' run_pgsr_openmaterial.sh

# 然后就可以使用旧脚本
bash run_neus2_openmaterial.sh 0 50 0
```

## ✨ 优势总结

### 模块化架构优势：

1. **无 Git 冲突**: 外部仓库独立管理，不会与主项目冲突
2. **统一接口**: 所有方法通过统一 API 调用
3. **易于扩展**: 添加新方法只需实现 wrapper
4. **更好的组织**: 清晰的目录结构
5. **灵活部署**: 可以选择性安装方法
6. **版本控制**: 每个外部仓库独立更新

### 与嵌入式相比：

| 特性 | 嵌入式 | 模块化 |
|------|--------|--------|
| Git 管理 | 复杂（子模块） | 简单（独立） |
| 扩展性 | 困难 | 容易 |
| 接口统一 | 无 | 有 |
| 代码复用 | 低 | 高 |
| 维护成本 | 高 | 低 |

## 🔗 相关文档

- 方法 API 文档: `methods/README.md`（待创建）
- 旧部署指南: `SERVER_DEPLOYMENT.md`
- 快速参考: `QUICK_REFERENCE.md`

---

**注意**: 模块化重构是非破坏性的，旧的脚本仍可使用（需更新路径）。
