# 服务器部署完整指南

本文档详细说明如何在服务器上部署和运行 NeuS2、2DGS、PGSR 三个方法。

## 前置条件检查

### 1. 服务器硬件要求

```bash
# 检查 GPU
nvidia-smi

# 建议配置：
# - GPU: RTX 3090 / A100 或更好
# - 内存: 32GB+
# - 磁盘空间: 500GB+（用于数据集和输出）
```

### 2. 软件环境要求

```bash
# 检查 CUDA 版本（需要 >= 11.0）
nvcc --version

# 检查 Python（需要 3.8+）
python --version

# 检查 Conda
conda --version

# 检查 CMake（NeuS2 需要 >= 3.18）
cmake --version
```

如果 CMake 版本过低：
```bash
pip install cmake --upgrade
cmake --version  # 验证
```

## 步骤 1: 上传代码到服务器

### 方案 A: 使用 Git（推荐）

```bash
# 在服务器上
cd /path/to/your/workspace
git clone YOUR_REPO_URL OpenMaterial
cd OpenMaterial

# 克隆子仓库（如果还没有）
git clone --recursive https://github.com/19reborn/NeuS2.git
git clone --recursive https://github.com/hbb1/2d-gaussian-splatting.git 2DGS
git clone --recursive https://github.com/zju3dv/PGSR.git
```

### 方案 B: 使用 SCP/SFTP

```bash
# 在本地机器上
cd /home/fangsuo/py
tar -czf OpenMaterial.tar.gz OpenMaterial/
scp OpenMaterial.tar.gz user@server:/path/to/workspace/

# 在服务器上
cd /path/to/workspace
tar -xzf OpenMaterial.tar.gz
cd OpenMaterial
```

### 方案 C: 使用 rsync（推荐，支持增量同步）

```bash
# 在本地机器上
rsync -avz --progress \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    /home/fangsuo/py/OpenMaterial/ \
    user@server:/path/to/workspace/OpenMaterial/
```

## 步骤 2: 下载 OpenMaterial 数据集

```bash
cd OpenMaterial

# 获取 Hugging Face token
# 访问: https://huggingface.co/settings/tokens

# 方案 A: 下载消融数据集（用于快速测试，~10GB）
python download.py --token YOUR_HF_TOKEN --type ablation

# 方案 B: 下载完整数据集（~500GB）
python download.py --token YOUR_HF_TOKEN --type all

# 方案 C: 下载特定材质
python download.py --token YOUR_HF_TOKEN --type conductor
```

**验证数据集：**
```bash
ls -lh datasets/openmaterial/
# 应该看到多个物体目录
```

## 步骤 3: 设置 Conda 环境

### 3.1 创建 NeuS2 环境

```bash
cd NeuS2

# 创建环境
conda create -n neus2 python=3.9 -y
conda activate neus2

# 安装基础依赖
pip install -r requirements.txt

# 安装 PyTorch（根据您的 CUDA 版本调整）
# CUDA 11.8:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
# pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121

# 安装 PyTorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# 编译 CUDA 代码
cmake . -B build
cmake --build build --config RelWithDebInfo -j

# 验证编译成功
ls -lh build/testbed
# 应该看到 testbed 可执行文件

cd ..
```

**如果编译失败：**
```bash
# 清理重新编译
cd NeuS2
rm -rf build
cmake . -B build -DCMAKE_CUDA_ARCHITECTURES=80  # 根据 GPU 架构调整
cmake --build build --config RelWithDebInfo -j 8
cd ..
```

### 3.2 创建 2DGS 环境

```bash
cd 2DGS

# 方案 A: 使用环境文件
conda env create --file environment.yml
conda activate surfel_splatting

# 方案 B: 手动创建（如果环境文件有问题）
conda create -n surfel_splatting python=3.8 -y
conda activate surfel_splatting
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install plyfile tqdm

# 编译 CUDA 扩展
pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn

# 验证安装
python -c "import torch; print(torch.cuda.is_available())"
python -c "import diff_surfel_rasterization"

cd ..
```

### 3.3 创建 PGSR 环境

```bash
cd PGSR

conda create -n pgsr python=3.8 -y
conda activate pgsr

# 安装 PyTorch（根据 CUDA 版本）
# CUDA 11.8:
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt

# 编译 CUDA 扩展
pip install submodules/diff-plane-rasterization
pip install submodules/simple-knn

# 验证安装
python -c "import torch; print(torch.cuda.is_available())"

cd ..
```

## 步骤 4: 测试单个场景

在批量运行之前，先测试单个场景确保一切正常。

### 4.1 测试 NeuS2

```bash
# 找一个测试场景
TEST_SCENE=$(ls -d datasets/openmaterial/*/* | head -1)
echo "Testing with: $TEST_SCENE"

# 转换数据格式
conda activate neus2
python NeuS2/tools/convert_openmaterial.py \
    --input "$TEST_SCENE" \
    --output "test_neus2_data" \
    --splits train test

# 训练（5分钟测试）
cd NeuS2
python scripts/run.py \
    --scene ../test_neus2_data/transforms_train.json \
    --name test_run \
    --network dtu.json \
    --n_steps 1000 \
    --save_mesh \
    --save_mesh_path ../test_mesh.ply

cd ..

# 检查输出
ls -lh NeuS2/output/test_run/
ls -lh test_mesh.ply
```

### 4.2 测试 2DGS

```bash
conda activate surfel_splatting

cd 2DGS

TEST_SCENE=$(ls -d ../datasets/openmaterial/*/* | head -1)

# 训练（测试 1000 iterations）
python train.py \
    -s "$TEST_SCENE" \
    -m ../test_2dgs_output \
    --iterations 1000 \
    --lambda_normal 0.05 \
    --lambda_distortion 1000

# 提取 mesh
python render.py \
    -s "$TEST_SCENE" \
    -m ../test_2dgs_output \
    --skip_test --skip_train \
    --mesh_res 512

cd ..

# 检查输出
ls -lh test_2dgs_output/
```

### 4.3 测试 PGSR

```bash
conda activate pgsr

cd PGSR

TEST_SCENE=$(ls -d ../datasets/openmaterial/*/* | head -1)

# 训练（测试 1000 iterations）
python train.py \
    -s "$TEST_SCENE" \
    -m ../test_pgsr_output \
    --iterations 1000 \
    --gpu 0

# 提取 mesh
python render.py \
    -s "$TEST_SCENE" \
    -m ../test_pgsr_output \
    --iteration 1000 \
    --compute_mesh

cd ..

# 检查输出
ls -lh test_pgsr_output/
```

## 步骤 5: 批量运行

### 5.1 创建运行脚本

创建一个主控脚本 `run_all.sh`：

```bash
cat > run_all.sh << 'EOF'
#!/bin/bash

# 配置
START=0
END=50  # 调整为您要处理的物体数量
GPU_NEUS2=0
GPU_2DGS=1
GPU_PGSR=2

# 创建日志目录
mkdir -p logs

echo "========================================="
echo " Starting OpenMaterial Benchmark"
echo "========================================="
echo "Processing objects: $START - $END"
echo "GPUs: NeuS2=$GPU_NEUS2, 2DGS=$GPU_2DGS, PGSR=$GPU_PGSR"
echo ""

# 运行 NeuS2
echo "[$(date)] Starting NeuS2..."
nohup bash run_neus2_openmaterial.sh $START $END $GPU_NEUS2 > logs/neus2.log 2>&1 &
NEUS2_PID=$!

# 运行 2DGS
echo "[$(date)] Starting 2DGS..."
nohup bash run_2dgs_openmaterial.sh $START $END $GPU_2DGS > logs/2dgs.log 2>&1 &
TWOGGS_PID=$!

# 运行 PGSR
echo "[$(date)] Starting PGSR..."
nohup bash run_pgsr_openmaterial.sh $START $END $GPU_PGSR > logs/pgsr.log 2>&1 &
PGSR_PID=$!

echo ""
echo "All methods started!"
echo "NeuS2 PID: $NEUS2_PID"
echo "2DGS PID: $TWOGGS_PID"
echo "PGSR PID: $PGSR_PID"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/neus2.log"
echo "  tail -f logs/2dgs.log"
echo "  tail -f logs/pgsr.log"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
EOF

chmod +x run_all.sh
```

### 5.2 运行批量任务

```bash
# 运行所有方法
./run_all.sh

# 监控进度
tail -f logs/neus2.log  # 在另一个终端窗口
tail -f logs/2dgs.log
tail -f logs/pgsr.log

# 监控 GPU
watch -n 1 nvidia-smi
```

### 5.3 如果只有单 GPU

如果只有一块 GPU，顺序运行：

```bash
# 顺序运行
bash run_neus2_openmaterial.sh 0 50 0
bash run_2dgs_openmaterial.sh 0 50 0
bash run_pgsr_openmaterial.sh 0 50 0
```

## 步骤 6: 监控和管理

### 6.1 检查进程状态

```bash
# 查看运行中的进程
ps aux | grep "python.*train.py"

# 查看特定方法的进程
ps aux | grep neus2
ps aux | grep 2dgs
ps aux | grep pgsr
```

### 6.2 查看输出结构

```bash
# 查看生成的 mesh
ls -lh Mesh/neus2/meshes/
ls -lh Mesh/2dgs/meshes/
ls -lh Mesh/pgsr/meshes/

# 查看训练输出
ls -lh neus2_output/
ls -lh 2dgs_output/
ls -lh pgsr_output/
```

### 6.3 磁盘空间管理

```bash
# 检查磁盘使用
df -h .

# 检查各目录大小
du -sh Mesh/*
du -sh *_output/

# 如果空间不足，可以清理中间文件
# 注意：只在确认结果正确后删除
rm -rf NeuS2/output/*/checkpoints/*  # 保留最终 checkpoint
rm -rf 2DGS_output/*/point_cloud/iteration_*/  # 保留最终结果
```

## 步骤 7: 评估结果

### 7.1 等待所有任务完成

```bash
# 检查是否还有任务在运行
ps aux | grep "train.py"

# 等待所有任务完成
wait
```

### 7.2 运行评估

```bash
# 评估所有方法
bash evaluate_all_methods.sh

# 对比结果
python compare_methods.py --methods instant-nsr-pl-wmask neus2 2dgs pgsr

# 查看对比结果
cat method_comparison.csv
```

## 步骤 8: 下载结果到本地

```bash
# 在本地机器上

# 下载对比结果
scp user@server:/path/to/OpenMaterial/method_comparison.csv .

# 下载 mesh（可能很大）
rsync -avz --progress \
    user@server:/path/to/OpenMaterial/Mesh/ \
    ./OpenMaterial_Results/Mesh/

# 下载日志
scp -r user@server:/path/to/OpenMaterial/logs/ ./logs/
```

## 故障排除

### 问题 1: CUDA Out of Memory

```bash
# 方案 A: 减少批次大小
# 编辑各方法的配置文件，降低 batch size

# 方案 B: 降低分辨率
# NeuS2: 降低 marching_cubes_res
# 2DGS/PGSR: 降低 mesh_res

# 方案 C: 分批处理
bash run_neus2_openmaterial.sh 0 25 0
bash run_neus2_openmaterial.sh 25 50 0
```

### 问题 2: 编译失败

```bash
# NeuS2 编译失败
cd NeuS2
rm -rf build
git submodule update --init --recursive
cmake . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4  # 减少并行数

# 2DGS/PGSR 扩展安装失败
cd 2DGS  # 或 PGSR
git submodule update --init --recursive
pip install --upgrade pip setuptools wheel
pip install submodules/diff-surfel-rasterization --no-cache-dir
```

### 问题 3: 数据集下载失败

```bash
# 重试下载
python download.py --token YOUR_TOKEN --type ablation

# 如果 Hugging Face 访问慢，使用镜像
export HF_ENDPOINT=https://hf-mirror.com
python download.py --token YOUR_TOKEN --type ablation
```

### 问题 4: 任务意外终止

```bash
# 查看日志
tail -100 logs/neus2.log
tail -100 logs/2dgs.log
tail -100 logs/pgsr.log

# 从断点继续
# 检查已完成的对象
ls Mesh/neus2/meshes/ | wc -l

# 从未完成的对象开始
bash run_neus2_openmaterial.sh 25 50 0  # 如果前25个完成了
```

## 性能优化建议

### 1. 使用 tmux/screen

```bash
# 安装 tmux
sudo apt-get install tmux  # 或 yum install tmux

# 创建会话
tmux new -s openmaterial

# 运行任务
./run_all.sh

# 分离会话: Ctrl+B, 然后按 D
# 重新连接: tmux attach -t openmaterial
```

### 2. 优先级调整

```bash
# 如果某个方法更重要，提高其优先级
nice -n -10 bash run_neus2_openmaterial.sh 0 50 0 &
nice -n 0 bash run_2dgs_openmaterial.sh 0 50 1 &
nice -n 10 bash run_pgsr_openmaterial.sh 0 50 2 &
```

### 3. 分阶段运行

```bash
# 阶段 1: 快速方法（NeuS2）
bash run_neus2_openmaterial.sh 0 100 0

# 阶段 2: 中速方法（2DGS）
bash run_2dgs_openmaterial.sh 0 100 0

# 阶段 3: 慢速方法（PGSR）
bash run_pgsr_openmaterial.sh 0 100 0
```

## 预期时间表

假设使用 3 块 RTX 3090，处理 100 个物体（每个物体 ~5 个场景）：

| 方法 | 单场景时间 | 总场景数 | 预计总时间 |
|------|-----------|---------|-----------|
| NeuS2 | 5-10分钟 | ~500 | ~40-80小时 |
| 2DGS | 15-20分钟 | ~500 | ~120-160小时 |
| PGSR | 20-30分钟 | ~500 | ~160-240小时 |

**并行运行（3 GPU）**: 最长方法的时间，约 **160-240小时（7-10天）**

## 检查清单

部署前确认：
- [ ] 服务器 GPU 可用（nvidia-smi）
- [ ] CUDA 版本 >= 11.0
- [ ] CMake 版本 >= 3.18（NeuS2）
- [ ] 磁盘空间 > 500GB
- [ ] OpenMaterial 数据集已下载
- [ ] 所有 conda 环境已创建
- [ ] NeuS2 已成功编译
- [ ] 2DGS/PGSR CUDA 扩展已安装
- [ ] 单场景测试成功
- [ ] 日志目录已创建
- [ ] 使用 tmux/screen 避免连接断开

## 完成后

1. ✅ 验证所有 mesh 已生成
2. ✅ 运行评估脚本
3. ✅ 生成对比报告
4. ✅ 备份重要结果
5. ✅ 清理中间文件（如需要）
6. ✅ 下载结果到本地

祝运行顺利！
