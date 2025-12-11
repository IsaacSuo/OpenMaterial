# 从 CPU 到 GPU：TSDF 三维重建加速完全指南

> 本文档面向零基础读者，从最基本的概念讲起，记录了将 TSDF 从 CPU 迁移到 GPU 的完整过程。

---

## 目录

1. [背景知识：什么是三维重建？](#1-背景知识什么是三维重建)
2. [核心概念：TSDF 是什么？](#2-核心概念tsdf-是什么)
3. [为什么要从 CPU 迁移到 GPU？](#3-为什么要从-cpu-迁移到-gpu)
4. [Open3D 的两套 API](#4-open3d-的两套-api)
5. [GPU TSDF 的核心数据结构](#5-gpu-tsdf-的核心数据结构)
6. [最难的部分：设备与类型约束](#6-最难的部分设备与类型约束)
7. [完整代码实现](#7-完整代码实现)
8. [踩坑记录与解决方案](#8-踩坑记录与解决方案)
9. [参数调优指南](#9-参数调优指南)
10. [总结与心得](#10-总结与心得)

---

## 1. 背景知识：什么是三维重建？

### 1.1 从照片到 3D 模型

想象一下：你用手机围着一个雕塑拍了 50 张照片，然后想把它变成一个 3D 模型（可以在电脑里 360° 旋转查看）。这个过程就叫做 **三维重建**。

```
📷 多张照片 → 🔄 某种算法 → 🗿 3D 模型 (Mesh)
```

### 1.2 什么是 Mesh？

**Mesh（网格）** 是 3D 模型最常见的表示方式：

- 由很多小三角形拼接而成
- 就像用很多小纸片折成一个物体
- 三角形越多、越小，模型越精细

```
粗糙的球：用 20 个三角形 → 看起来像多面体
精细的球：用 20000 个三角形 → 看起来很光滑
```

### 1.3 重建流程概述

本项目使用 **2DGS（2D Gaussian Splatting）** 进行三维重建，流程如下：

```
输入图片 → 2DGS 训练 → 得到高斯点云 → 渲染深度图 → TSDF 融合 → 提取 Mesh
                                            ↑
                                       我们优化的部分
```

---

## 2. 核心概念：TSDF 是什么？

### 2.1 问题：如何从深度图得到 Mesh？

**深度图（Depth Map）** 是一张特殊的图片：
- 普通照片：每个像素存储颜色 (R, G, B)
- 深度图：每个像素存储"这个点离相机多远"

```
深度图示例（数值代表距离，单位米）：
┌─────────────────────┐
│ 2.1  2.0  2.0  2.1  │  ← 背景墙，距离 2 米
│ 1.5  1.2  1.2  1.5  │  ← 人的肩膀
│ 1.5  1.0  1.0  1.5  │  ← 人的身体
│ 1.8  1.8  1.8  1.8  │  ← 地面
└─────────────────────┘
```

我们有 50 张从不同角度拍的深度图，如何把它们"融合"成一个完整的 3D 模型？

### 2.2 TSDF 的直觉理解

**TSDF = Truncated Signed Distance Function（截断符号距离函数）**

想象把整个 3D 空间切成很多很多小方块（体素，Voxel），就像乐高积木一样：

```
┌───┬───┬───┬───┐
│   │   │   │   │
├───┼───┼───┼───┤
│   │ ● │ ● │   │  ← 物体表面经过这些格子
├───┼───┼───┼───┤
│   │   │   │   │
└───┴───┴───┴───┘
```

对于每个小方块，我们记录一个数值：**它离最近的物体表面有多远**

- 正数：在物体外面（空气中）
- 负数：在物体里面
- 零：刚好在表面上

```
TSDF 值示意图：

+2   +1   +0.5   0   -0.5  -1   -2
空气 ←───────── 表面 ──────────→ 物体内部
```

### 2.3 为什么叫"截断"？

我们只关心表面附近的区域，太远的地方直接忽略：

```
             截断距离
         ←────────────→
    +∞   +1   0   -1   -∞
         ↓       ↓
    只保留这个范围内的值
```

这样可以节省存储空间，也让算法更稳定。

### 2.4 多帧融合

每张深度图会更新一部分体素的 TSDF 值。多张图融合后：

```
第 1 帧：从正面看，更新正面的体素
第 2 帧：从侧面看，更新侧面的体素
第 3 帧：从背面看，更新背面的体素
...
第 50 帧：所有角度都覆盖了，得到完整的 TSDF 体积
```

### 2.5 提取 Mesh

最后，用 **Marching Cubes** 算法从 TSDF 体积中提取出表面：

```
找到所有 TSDF = 0 的位置 → 连接成三角形 → 得到 Mesh
```

---

## 3. 为什么要从 CPU 迁移到 GPU？

### 3.1 CPU 版本的问题

原来使用 Open3D 的 CPU 版本：

```python
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.004,  # 4mm 的体素
    sdf_trunc=0.02,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8Integrated
)
```

**问题 1：内存爆炸（OOM = Out Of Memory）**

场景大小：2.4m × 2.4m × 2.4m（花园场景）

```
体素数量 = (2.4 / 0.004)³ = 600³ = 2.16 亿个体素
每个体素约 20 字节
总内存 = 2.16 亿 × 20 = 43.2 亿字节 ≈ 4.3 GB
```

实际上 CPU 版本不是真正的稀疏存储，内存占用更高，直接 OOM。

**问题 2：速度慢**

每帧深度图集成需要 3-5 秒，50 帧就是 2-4 分钟。

### 3.2 GPU 版本的优势

| 对比项 | CPU 版本 | GPU 版本 |
|--------|----------|----------|
| 存储方式 | 密集数组 | 稀疏哈希表 |
| 内存类型 | RAM | VRAM（显存）|
| 速度 | 慢 | 快 5-10 倍 |
| 小体素支持 | 容易 OOM | 可以更小 |

**稀疏存储的威力：**

```
密集存储：必须存储所有 2.16 亿个体素（大部分是空气）
稀疏存储：只存储表面附近的体素（可能只有几百万个）

实际节省：从 4.3GB → 几百 MB
```

---

## 4. Open3D 的两套 API

Open3D 提供了两套完全不同的 API：

### 4.1 传统 API（CPU）

```python
import open3d as o3d

# 传统几何体
mesh = o3d.geometry.TriangleMesh()
pcd = o3d.geometry.PointCloud()

# 传统 TSDF
volume = o3d.pipelines.integration.ScalableTSDFVolume(...)
```

特点：
- 简单易用
- 只支持 CPU
- 文档完善

### 4.2 Tensor API（CPU/GPU）

```python
import open3d as o3d
import open3d.core as o3c  # 核心张量模块

# Tensor 几何体
mesh_t = o3d.t.geometry.TriangleMesh()
pcd_t = o3d.t.geometry.PointCloud()

# Tensor TSDF
vbg = o3d.t.geometry.VoxelBlockGrid(...)  # 支持 GPU！
```

特点：
- 更复杂
- 支持 CPU 和 GPU
- 文档不完善，很多坑

### 4.3 两套 API 的转换

```python
# Tensor → 传统
mesh_legacy = mesh_tensor.to_legacy()

# 传统 → Tensor
mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
```

---

## 5. GPU TSDF 的核心数据结构

### 5.1 VoxelBlockGrid 详解

```python
self.vbg = o3d.t.geometry.VoxelBlockGrid(
    attr_names=('tsdf', 'weight', 'color'),      # 每个体素存什么
    attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),  # 数据类型
    attr_channels=((1), (1), (3)),               # 每个属性几个通道
    voxel_size=0.005,                            # 体素大小（米）
    block_resolution=16,                          # 每个 block 的分辨率
    block_count=100000,                          # 哈希表容量
    device=o3c.Device("CUDA:0")                  # 放在 GPU 上
)
```

### 5.2 Block 是什么？

为了高效管理，体素被组织成 **Block（块）**：

```
一个 Block = 16 × 16 × 16 = 4096 个体素

┌─────────────────────────────────────────────┐
│  Block 0    Block 1    Block 2    ...       │
│ ┌───────┐  ┌───────┐  ┌───────┐             │
│ │16×16×16│ │16×16×16│ │16×16×16│            │
│ │ voxels │ │ voxels │ │ voxels │            │
│ └───────┘  └───────┘  └───────┘             │
└─────────────────────────────────────────────┘
```

**为什么要用 Block？**

- 方便做"视锥剔除"（只处理相机能看到的区域）
- 方便稀疏存储（只分配有物体的 Block）
- GPU 并行处理的基本单位

### 5.3 block_count 的含义

`block_count=100000` 是**哈希表的容量**，不是实际分配的显存！

```
打个比方：
- block_count = 通讯录能存多少人
- 实际 block 数 = 你真正加了多少好友

通讯录能存 10 万人，不代表你真的有 10 万好友
```

**显存估算公式：**

```
实际显存 ≈ 实际使用的 block 数 × 16³ × 每体素字节数
         = 实际 block 数 × 4096 × 20 bytes

例：66000 blocks × 4096 × 20 ≈ 5.4 GB
```

### 5.4 每个体素存什么？

```python
attr_names=('tsdf', 'weight', 'color')
attr_channels=((1), (1), (3))
```

| 属性 | 通道数 | 含义 |
|------|--------|------|
| tsdf | 1 | 符号距离值 |
| weight | 1 | 置信度（被多少帧观测过）|
| color | 3 | RGB 颜色 |

每个体素总共：(1 + 1 + 3) × 4 bytes = 20 bytes

---

## 6. 最难的部分：设备与类型约束

这是整个迁移过程中**最坑**的部分。Open3D 的 GPU TSDF 对每个参数有严格的要求，但文档几乎没有说明。

### 6.1 设备约束总结

| 参数 | 必须在哪里 | 为什么 |
|------|-----------|--------|
| depth_img | GPU | 图像数据量大，GPU 处理快 |
| color_img | GPU | 同上 |
| block_coords | GPU | 索引数组，GPU 并行访问 |
| intrinsic | **CPU** | C++ 内核配置需要从 CPU 读取 |
| extrinsic | **CPU** | 同上 |

### 6.2 类型约束总结

| 参数 | 必须的类型 | 常见错误 |
|------|-----------|----------|
| depth_img | Float32 | - |
| color_img | **Float32** | 原始是 UInt8，必须转换！|
| block_coords | **Int32** | 默认是 Int64，必须转换！|
| intrinsic | Float64 | - |
| extrinsic | Float64 | - |
| depth_scale | Python float | 不能是 numpy 类型 |
| depth_max | Python float | 不能是 numpy 类型 |

### 6.3 为什么 intrinsic/extrinsic 必须在 CPU？

这是最反直觉的一点。直觉上，GPU 计算应该所有数据都在 GPU 上才对。

**真相**（通过阅读 Open3D C++ 源码发现）：

```cpp
// Open3D 内部代码 (简化)
void integrate(...) {
    // 1. 在 CPU 上读取相机矩阵
    float fx = intrinsic[0][0];  // 从 CPU 内存读取
    float fy = intrinsic[1][1];

    // 2. 用这些值配置 CUDA kernel
    kernel<<<blocks, threads>>>(fx, fy, ...);

    // 3. kernel 在 GPU 上并行处理图像
}
```

如果 intrinsic 在 GPU 上，第 1 步就会报错：CPU 无法直接读取 GPU 内存！

---

## 7. 完整代码实现

### 7.1 类定义与初始化

```python
import open3d as o3d
import open3d.core as o3c

class GPUTSDFVolume:
    def __init__(self, voxel_length=0.004, sdf_trunc=None, device='cuda:0'):
        self.voxel_size = voxel_length
        self.sdf_trunc = sdf_trunc if sdf_trunc else 4.0 * voxel_length

        # 设置设备
        self.device = o3c.Device(device)

        # 初始化 VoxelBlockGrid
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=self.voxel_size,
            block_resolution=16,
            block_count=100000,  # 哈希表容量
            device=self.device
        )
```

### 7.2 integrate 方法（核心）

```python
def integrate(self, rgbd, intrinsic, extrinsic):
    """
    将一帧 RGBD 图像融合到 TSDF 体积中

    参数：
        rgbd: o3d.t.geometry.RGBDImage (GPU tensor)
        intrinsic: 相机内参矩阵 3×3 (GPU tensor)
        extrinsic: 相机外参矩阵 4×4 (GPU tensor)
    """

    # ========== 第 1 步：准备图像数据 ==========
    # 图像必须在 GPU 上，且为 Float32
    depth_img = rgbd.depth
    color_img = rgbd.color.to(o3c.Dtype.Float32)  # UInt8 → Float32

    # ========== 第 2 步：准备相机矩阵 ==========
    # 关键：矩阵必须在 CPU 上！
    cpu_device = o3c.Device("CPU:0")
    intrinsic_cpu = intrinsic.to(cpu_device)
    extrinsic_cpu = extrinsic.to(cpu_device)

    # ========== 第 3 步：计算深度参数 ==========
    depth_np = depth_img.as_tensor().cpu().numpy()
    depth_scale = 1.0
    depth_max = float(depth_np.max()) if depth_np.max() > 0 else 10.0

    # ========== 第 4 步：视锥剔除 ==========
    # 计算当前帧能看到哪些 blocks
    frustum_block_coords = self.vbg.compute_unique_block_coordinates(
        depth_img,
        intrinsic_cpu,
        extrinsic_cpu,
        depth_scale=depth_scale,
        depth_max=depth_max,
        trunc_voxel_multiplier=4.0
    )

    # ========== 第 5 步：转换 block 坐标 ==========
    # 必须：GPU + Int32
    gpu_device = o3c.Device("CUDA:0")
    frustum_block_coords = frustum_block_coords.to(gpu_device)
    frustum_block_coords = frustum_block_coords.to(o3c.int32)  # Int64 → Int32

    # ========== 第 6 步：执行融合 ==========
    self.vbg.integrate(
        frustum_block_coords,  # GPU, Int32
        depth_img,             # GPU, Float32
        color_img,             # GPU, Float32
        intrinsic_cpu,         # CPU, Float64  ← 注意！
        extrinsic_cpu,         # CPU, Float64  ← 注意！
        float(depth_scale),    # Python float
        float(depth_max)       # Python float
    )
```

### 7.3 提取 Mesh

```python
def extract_triangle_mesh(self):
    """从 TSDF 体积中提取三角网格"""

    # 使用 Marching Cubes 算法提取
    mesh_tensor = self.vbg.extract_triangle_mesh()

    # 转换为传统格式（兼容其他代码）
    mesh_legacy = mesh_tensor.to_legacy()

    return mesh_legacy
```

---

## 8. 踩坑记录与解决方案

### 8.1 坑 1：ParallelFor 错误

**错误信息：**
```
RuntimeError: ParallelFor for CUDA cannot run on device CPU:0
```

**原因：** 把 CPU tensor 传给了需要 GPU tensor 的参数

**解决：** 检查所有图像数据是否在 GPU 上

```python
# 错误
depth_img = o3c.Tensor(depth_np, device=o3c.Device("CPU:0"))

# 正确
depth_img = o3c.Tensor(depth_np, device=o3c.Device("CUDA:0"))
```

### 8.2 坑 2：InverseTransformation 错误

**错误信息：**
```
RuntimeError: InverseTransformation expected CPU:0 but got CUDA:0
```

**原因：** 相机矩阵必须在 CPU 上

**解决：**
```python
intrinsic_cpu = intrinsic.to(o3c.Device("CPU:0"))
extrinsic_cpu = extrinsic.to(o3c.Device("CPU:0"))
```

### 8.3 坑 3：类型不匹配错误

**错误信息：**
```
TypeError: incompatible function arguments
```

**可能原因 1：** block_coords 是 Int64 而不是 Int32

```python
# 修复
frustum_block_coords = frustum_block_coords.to(o3c.int32)
```

**可能原因 2：** color 是 UInt8 而不是 Float32

```python
# 修复
color_img = rgbd.color.to(o3c.Dtype.Float32)
```

**可能原因 3：** depth_scale/depth_max 是 numpy 类型

```python
# 错误
depth_max = depth_np.max()  # 返回 numpy.float64

# 正确
depth_max = float(depth_np.max())  # 转为 Python float
```

### 8.4 坑 4：.to() 方法不能同时传两个参数

**错误信息：**
```
TypeError: to(): incompatible function arguments
```

**错误写法：**
```python
tensor = tensor.to(device=gpu_device, dtype=o3c.int32)
```

**正确写法：**
```python
tensor = tensor.to(gpu_device)  # 先转设备
tensor = tensor.to(o3c.int32)   # 再转类型
```

### 8.5 坑 5：哈希表溢出

**错误信息：**
```
stdgpu::vector::size : Size out of bounds. Resizing to 0
```

**原因：** 实际需要的 block 数超过了 `block_count` 容量

**解决：** 增大 `block_count` 或增大 `voxel_size`

```python
# 方案 1：增大容量
block_count=200000  # 从 100000 增加到 200000

# 方案 2：增大体素（减少 block 数）
voxel_size=0.006  # 从 0.004 增加到 0.006
```

### 8.6 坑 6：初始化时 OOM

**错误信息：**
```
RuntimeError: CUDA runtime error: out of memory
```

**原因：** `block_count` 设置太大，哈希表结构本身占用显存

**解决：** 减小 `block_count`

```python
# block_count 本身也占显存（虽然是稀疏的，但哈希表结构有开销）
# 500000 可能太大，尝试 100000
block_count=100000
```

---

## 9. 参数调优指南

### 9.1 voxel_size（体素大小）

| 值 | 效果 | 适用场景 |
|----|------|----------|
| 0.002 | 极高精度，block 数爆炸 | 小物体、高端显卡 |
| 0.004 | 高精度 | 中等物体 |
| 0.005 | 平衡 | 推荐起始值 |
| 0.01 | 较低精度，很稳定 | 大场景、调试用 |

**经验公式：**
```
预估 block 数 ≈ (场景尺寸 / voxel_size / 16)³ × 稀疏系数
稀疏系数通常 0.1-0.3（表面占总体积的比例）
```

### 9.2 sdf_trunc（SDF 截断距离）

通常设为 `4-5 × voxel_size`

```python
sdf_trunc = 5.0 * voxel_size  # 推荐
```

太小：表面可能有孔洞
太大：表面可能变厚、模糊

### 9.3 depth_trunc（深度截断）

超过这个距离的深度值会被忽略

```python
depth_trunc = 5.0  # 5 米，适合室内场景
depth_trunc = 10.0 # 10 米，适合室外场景
```

### 9.4 block_count（哈希表容量）

| 场景大小 | 建议值 |
|----------|--------|
| 小物体（< 1m）| 50000 |
| 中等场景（1-3m）| 100000 |
| 大场景（> 3m）| 150000-200000 |

**调试技巧：** 打印实际使用的 block 数

```python
total_blocks = self.vbg.hashmap().size()
print(f"Total blocks: {total_blocks} / {block_count}")
```

---

## 10. 总结与心得

### 10.1 这次迁移的核心挑战

1. **不是算法，是 API**
   - TSDF 算法本身很成熟
   - 难的是 Open3D Tensor API 的各种隐式约束

2. **文档不完善**
   - 官方文档只有简单示例
   - 很多约束要看 C++ 源码或试错

3. **CPU/GPU 混合使用**
   - 不是所有数据都放 GPU 就行
   - 某些参数必须在 CPU（反直觉！）

### 10.2 调试方法论

1. **先用大 voxel_size 跑通流程**
   - 从 0.01 开始，确保代码逻辑正确
   - 再逐步减小到 0.005、0.004

2. **打印中间状态**
   - 打印 tensor 的 device、dtype、shape
   - 打印 block 数量监控容量

3. **看报错信息的关键词**
   - `CPU:0` vs `CUDA:0` → 设备问题
   - `Float32` vs `Int64` → 类型问题
   - `out of bounds` → 容量问题

### 10.3 最终收获

成功实现了 GPU 加速的 TSDF 融合：
- 速度提升 5-10 倍
- 可以使用更小的 voxel_size
- 不再受 RAM 限制（改用 VRAM）

这为后续的网格质量优化（Grid Search 调参）打下了基础。

---

## 附录：关键文件路径

```
OpenMaterial/
├── methods/utils/gpu_tsdf.py      # GPU TSDF 实现
├── external/2DGS/utils/mesh_utils.py  # 调用 TSDF 的地方
├── grid_search_2dgs_stage1.py     # 网格搜索脚本
└── grid_search_2dgs_stage2.py     # 网格搜索脚本
```

---

*最后更新：2024年12月*
*作者：基于 Claude 与用户的协作调试记录整理*
