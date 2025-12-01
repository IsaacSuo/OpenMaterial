# 快速参考卡片

## 一、上传到服务器

```bash
# 方法 1: rsync（推荐）
rsync -avz --progress /home/fangsuo/py/OpenMaterial/ user@server:/path/to/OpenMaterial/

# 方法 2: Git
git push origin master
# 在服务器上: git pull
```

## 二、环境配置（三个环境）

```bash
# === NeuS2 环境 ===
conda create -n neus2 python=3.9 -y
conda activate neus2
cd NeuS2
pip install -r requirements.txt
pip install torch torchvision pytorch3d
cmake . -B build && cmake --build build --config RelWithDebInfo -j
cd ..

# === 2DGS 环境 ===
conda create -n surfel_splatting python=3.8 -y
conda activate surfel_splatting
cd 2DGS
pip install torch torchvision plyfile tqdm
pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn
cd ..

# === PGSR 环境 ===
conda create -n pgsr python=3.8 -y
conda activate pgsr
cd PGSR
pip install torch torchvision torchaudio -r requirements.txt
pip install submodules/diff-plane-rasterization
pip install submodules/simple-knn
cd ..
```

## 三、下载数据

```bash
# 测试用（小数据集）
python download.py --token YOUR_TOKEN --type ablation

# 完整数据集
python download.py --token YOUR_TOKEN --type all
```

## 四、运行命令

### 单 GPU 顺序运行

```bash
bash run_neus2_openmaterial.sh 0 50 0
bash run_2dgs_openmaterial.sh 0 50 0
bash run_pgsr_openmaterial.sh 0 50 0
```

### 多 GPU 并行运行

```bash
# 创建运行脚本
cat > run_parallel.sh << 'EOF'
#!/bin/bash
nohup bash run_neus2_openmaterial.sh 0 50 0 > logs/neus2.log 2>&1 &
nohup bash run_2dgs_openmaterial.sh 0 50 1 > logs/2dgs.log 2>&1 &
nohup bash run_pgsr_openmaterial.sh 0 50 2 > logs/pgsr.log 2>&1 &
EOF

chmod +x run_parallel.sh
mkdir -p logs
./run_parallel.sh
```

### 使用 tmux（防止断连）

```bash
tmux new -s openmaterial
./run_parallel.sh
# 按 Ctrl+B 然后 D 分离
# 重连: tmux attach -t openmaterial
```

## 五、监控

```bash
# 查看日志
tail -f logs/neus2.log
tail -f logs/2dgs.log
tail -f logs/pgsr.log

# 查看 GPU
watch -n 1 nvidia-smi

# 查看进程
ps aux | grep "train.py"

# 查看输出
ls -lh Mesh/*/meshes/
```

## 六、评估

```bash
# 等待完成后
bash evaluate_all_methods.sh
python compare_methods.py
cat method_comparison.csv
```

## 七、常用调试命令

```bash
# 测试单个场景
TEST_SCENE=$(ls -d datasets/openmaterial/*/* | head -1)

# NeuS2 测试
conda activate neus2
python NeuS2/tools/convert_openmaterial.py --input "$TEST_SCENE" --output test_data
cd NeuS2 && python scripts/run.py --scene ../test_data/transforms_train.json --name test --network dtu.json --n_steps 1000

# 2DGS 测试
conda activate surfel_splatting
cd 2DGS && python train.py -s "$TEST_SCENE" -m ../test_out --iterations 1000

# PGSR 测试
conda activate pgsr
cd PGSR && python train.py -s "$TEST_SCENE" -m ../test_out --iterations 1000
```

## 八、故障排除

### CMake 版本低

```bash
pip install cmake --upgrade
```

### CUDA OOM

```bash
# 减少 batch size 或降低分辨率
# 或分批运行:
bash run_neus2_openmaterial.sh 0 25 0
bash run_neus2_openmaterial.sh 25 50 0
```

### 编译失败

```bash
cd NeuS2  # 或 2DGS/PGSR
rm -rf build
git submodule update --init --recursive
# 重新编译
```

### 任务中断

```bash
# 查看已完成数量
ls Mesh/neus2/meshes/ | wc -l

# 从断点继续
bash run_neus2_openmaterial.sh 25 50 0  # 假设前25个完成
```

## 九、文件位置

| 文件 | 说明 |
|------|------|
| `SERVER_DEPLOYMENT.md` | 完整部署指南 |
| `METHODS_INTEGRATION.md` | 方法详细文档 |
| `QUICKSTART.md` | 快速开始 |
| `run_*_openmaterial.sh` | 各方法运行脚本 |
| `evaluate_all_methods.sh` | 统一评估 |
| `compare_methods.py` | 结果对比 |

## 十、预期时间

| 配置 | 处理 50 对象（~250 场景） |
|------|------------------------|
| 单 GPU 顺序 | ~30-50 天 |
| 3 GPU 并行 | ~10-15 天 |

## 完整流程一览

```bash
# 1. 上传代码
rsync -avz OpenMaterial/ server:/path/

# 2. 登录服务器
ssh server
cd /path/OpenMaterial

# 3. 下载数据
python download.py --token TOKEN --type ablation

# 4. 配置环境（运行一次）
# ... 见"二、环境配置"

# 5. 测试（可选）
# ... 见"七、调试命令"

# 6. 批量运行
tmux new -s openmaterial
./run_parallel.sh
# Ctrl+B, D

# 7. 监控（另一个窗口）
tail -f logs/*.log
watch nvidia-smi

# 8. 评估（完成后）
bash evaluate_all_methods.sh
python compare_methods.py

# 9. 下载结果
scp server:/path/method_comparison.csv .
```

---

💡 **提示**: 详细说明见 `SERVER_DEPLOYMENT.md`
