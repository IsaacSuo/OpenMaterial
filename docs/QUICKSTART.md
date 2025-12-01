# 模块化架构 - 快速开始

## 30秒上手

```bash
# 1. 设置仓库
python setup_methods.py --clone

# 2. 设置环境
python setup_methods.py --setup all

# 3. 运行基准测试
python run_benchmark.py --method all --start 0 --end 10 --gpus 0,1,2
```

## 核心改变

### 之前（嵌入式）

```
OpenMaterial/
├── NeuS2/          # ← Git 仓库嵌入，容易冲突
├── 2DGS/           # ← Git 仓库嵌入
├── PGSR/           # ← Git 仓库嵌入
└── run_*.sh        # ← 分散的脚本
```

### 现在（模块化）

```
OpenMaterial/
├── methods/           # ← 统一接口层
│   └── wrappers/      # ← 各方法的包装器
├── external/          # ← 外部仓库（独立管理）
│   ├── NeuS2/
│   ├── 2DGS/
│   └── PGSR/
├── setup_methods.py   # ← 统一设置脚本
└── run_benchmark.py   # ← 统一运行脚本
```

## 主要命令

### 1. 仓库管理（setup_methods.py）

```bash
# 克隆外部仓库
python setup_methods.py --clone

# 移动已有仓库（如果您之前 git clone 过）
python setup_methods.py --move
rm -rf NeuS2/ 2DGS/ PGSR/  # 清理旧目录

# 设置方法环境
python setup_methods.py --setup all           # 所有方法
python setup_methods.py --setup neus2         # 单个方法
python setup_methods.py --setup 2dgs pgsr     # 多个方法
```

### 2. 运行基准测试（run_benchmark.py）

```bash
# 单个方法
python run_benchmark.py --method neus2 --start 0 --end 50 --gpu 0

# 所有方法并行
python run_benchmark.py --method all --start 0 --end 50 --gpus 0,1,2

# 带自定义配置
python run_benchmark.py --method neus2 --config my_config.json --gpu 0

# 干运行（查看会做什么）
python run_benchmark.py --method neus2 --dry-run
```

## 从旧脚本迁移

### 旧方式
```bash
bash run_neus2_openmaterial.sh 0 50 0 &
bash run_2dgs_openmaterial.sh 0 50 1 &
bash run_pgsr_openmaterial.sh 0 50 2 &
wait
```

### 新方式
```bash
python run_benchmark.py --method all --start 0 --end 50 --gpus 0,1,2
```

## Python API 使用

```python
from methods import get_method

# 获取方法
neus2 = get_method('neus2')(repo_path='external/NeuS2')

# 处理场景
result = neus2.process_scene(
    input_scene='datasets/openmaterial/obj001/scene001',
    output_dir='output',
    gpu_id=0
)

print(result['mesh_output'])  # output/meshes/obj001/scene001.ply
```

## 配置文件示例

```json
// config.json
{
    "n_steps": 20000,
    "marching_cubes_res": 1024
}
```

```bash
python run_benchmark.py --method neus2 --config config.json --gpu 0
```

## 服务器部署

```bash
# 1. 上传（本地）
rsync -avz OpenMaterial/ user@server:/path/

# 2. 设置（服务器）
cd /path/OpenMaterial
python download.py --token TOKEN --type ablation
python setup_methods.py --clone
python setup_methods.py --setup all

# 3. 运行（使用 tmux）
tmux new -s bench
python run_benchmark.py --method all --start 0 --end 50 --gpus 0,1,2
# Ctrl+B, D 分离

# 4. 监控
tail -f benchmark_output/*/benchmark_results.json
```

## 常见问题

**Q: 我已经有 NeuS2/ 等目录，怎么办？**

```bash
python setup_methods.py --move
rm -rf NeuS2/ 2DGS/ PGSR/
```

**Q: 旧脚本还能用吗？**

可以，但需要更新路径：
```bash
sed -i 's|./NeuS2|./external/NeuS2|g' run_neus2_openmaterial.sh
```

**Q: 如何只设置一个方法？**

```bash
python setup_methods.py --clone  # 先克隆
python setup_methods.py --setup neus2  # 只设置 NeuS2
```

**Q: 如何查看详细错误？**

```bash
# 运行脚本会显示详细输出
python run_benchmark.py --method neus2 --start 0 --end 1 --gpu 0

# 或检查环境
conda activate neus2
python -c "import torch; print(torch.cuda.is_available())"
```

## 完整流程示例

```bash
# 克隆项目
git clone YOUR_REPO
cd OpenMaterial

# 下载数据
python download.py --token TOKEN --type ablation

# 设置方法
python setup_methods.py --clone
python setup_methods.py --setup all

# 测试一个场景
python run_benchmark.py \
    --method neus2 \
    --start 0 --end 1 \
    --gpu 0

# 运行完整基准测试
python run_benchmark.py \
    --method all \
    --start 0 --end 50 \
    --gpus 0,1,2

# 评估结果
bash evaluate_all_methods.sh
python compare_methods.py
```

## 优势

✅ 统一接口，简化使用
✅ 无 Git 子模块冲突
✅ 更好的错误处理
✅ 自动结果保存
✅ 支持并行运行
✅ 配置文件支持
✅ 易于扩展新方法

## 更多信息

- 详细文档: `MODULAR_DEPLOYMENT.md`
- API 文档: `methods/base_method.py`
- 原部署指南: `SERVER_DEPLOYMENT.md`
