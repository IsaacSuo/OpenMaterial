# 坐标系修复指南（快速开始）

## 问题
- `m_n ≈ 0.05`（不变）
- `m_d = nan`
- GT 法线和渲染法线不匹配

## 原因（一句话）
GT 法线未被 flip_yz 变换，与变换后的相机参考系不匹配。

## 修复
在 `scripts/render_gt_open3d.py` 第 163 行后添加：

```python
flip_yz_3x3 = np.array([[1,0,0], [0,-1,0], [0,0,-1]], dtype=np.float32)
normals = normals @ flip_yz_3x3.T
```

## 执行步骤

### 1. 备份原文件
```bash
cp scripts/render_gt_open3d.py scripts/render_gt_open3d.py.backup
```

### 2. 编辑文件
打开 `scripts/render_gt_open3d.py`，找到第 163 行：
```python
normals = np.asarray(mesh_normal.vertex_normals)
```

改为：
```python
normals = np.asarray(mesh_normal.vertex_normals)

# FIX: Apply coordinate system transformation
flip_yz_3x3 = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
], dtype=np.float32)
normals = normals @ flip_yz_3x3.T
```

### 3. 删除旧的 GT 数据
```bash
rm -rf <your_dataset>/depth_gt
rm -rf <your_dataset>/normal_gt
```

### 4. 重新生成 GT
```bash
python scripts/render_gt_open3d.py -s <your_dataset> --debug
```

### 5. 重新训练
```bash
python train_pbr.py -s <your_dataset> ...
```

## 预期改进
- m_n：从 ~0.05 → 0.3-0.7（显著提升）
- m_d：从 nan → 0.1-0.5（从失败到工作）
- 训练收敛性提升

## 验证
检查训练日志中：
- 第一次有法线监督时，m_n 应有一个**跳跃**
- 如果仍为 0，检查 GT 法线文件是否生成

## 详细分析
见：
- `COORD_SYSTEM_ROOT_CAUSE_ANALYSIS.md`（完整分析）
- `COORDINATE_SYSTEM_SUMMARY.md`（总结）

## 技术细节

### 为什么需要 flip_yz？
- 2DGS 使用 OpenGL 坐标系（Y上, Z后）
- GT 渲染器（Open3D）使用不同坐标系（Y下, Z前）
- 相机位置应用了 flip_yz 变换
- 法线向量也必须应用相同变换，否则不匹配

### 数学验证
```
flip_yz = [1  0  0]
          [0 -1  0]
          [0  0 -1]

法线 n = (x, y, z)
变换后 n' = (x, -y, -z)
```

这确保了即使坐标系改变，法线方向仍然正确相对于几何。

## 常见问题

### Q: 修复后 m_n 仍然很低？
**A:** 
1. 检查 GT 法线文件是否生成（看 normal_gt 文件夹）
2. 检查法线是否归一化（应该有 magnitude ≈ 1）
3. 检查网格法线方向（可能需要 flip）

### Q: 修复后 m_d 仍是 nan？
**A:**
1. 检查深度文件是否生成（看 depth_gt 文件夹）
2. 检查深度值范围（应该都 > 0）
3. 检查网格是否在相机视锥内

### Q: 可以自动应用修复吗？
**A:** 可以，运行 `APPLY_FIX.sh`（如果在 Linux/Mac）

## 文件对应关系

| 文件 | 问题 | 修复 |
|------|------|------|
| `gaussian_renderer/__init__.py` | 法线变换 | ✓ 已正确 |
| `scripts/render_gt_open3d.py` | **GT 法线未变换** | ← **修复这里** |
| `train_pbr.py` | 法线损失计算 | ✓ 已正确 |

## 相关文档
- 原始分析：`COORDINATE_SYSTEM_ANALYSIS.md`
- 修复细节：`FIXES_RENDER_GT_OPEN3D.py`
- 调试工具：`DEBUG_COORD_SYSTEM.py`

---

**总结：** 一个简单的 2 行代码修复解决坐标系不一致问题。
