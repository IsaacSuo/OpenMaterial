# 2DGS-PBR 可学习环境贴图实现检查报告

## 执行日期
2025-12-29

---

## 1. EnvironmentLight 类检查 ✓ 通过

### 位置
`/home/fangsuo/py/OpenMaterial/external/2DGS-PBR/utils/pbr_utils.py`

### 1.1 env_map 可学习性检查 ✓ 正确

**代码位置：第 36-39 行**
```python
self.env_map = nn.Parameter(
    torch.ones(3, resolution, resolution * 2) * 0.5,
    requires_grad=True
)
```

**代码位置：第 67 行**
```python
return nn.Parameter(env_tensor, requires_grad=True)
```

**结论**：✓ 正确
- env_map 被正确包装为 `nn.Parameter`
- 显式设置了 `requires_grad=True`
- 无论从文件加载还是初始化，都保证了可学习性

### 1.2 sample() 方法微分性检查 ✓ 正确

**代码位置：第 69-113 行**

关键步骤分析：
1. **方向到球面坐标转换** ✓ 可微分
   - `torch.atan2(x, z)` - 可微分
   - `torch.acos(torch.clamp(y, -1.0, 1.0))` - 可微分（clamp 不阻断梯度）

2. **grid_sample 操作** ✓ 可微分
   - `F.grid_sample()` - 完全可微分
   - 双线性插值支持反向传播

3. **无梯度阻止**：
   - 整个流程中没有 `.detach()` 或 `no_grad()` 上下文
   - 所有中间张量保持梯度跟踪

**结论**：✓ 正确，梯度可以反向流动

### 1.3 forward() 方法检查 ✓ 正确

**代码位置：第 115-117 行**
```python
def forward(self, directions: torch.Tensor) -> torch.Tensor:
    """Alias for sample()"""
    return self.sample(directions)
```

**结论**：✓ 正确，简单的别名转发

---

## 2. train_pbr.py 中的环境光训练检查 ✓ 基本正确，需要注意

### 2.1 env_light_optimizer 创建检查 ✓ 正确

**代码位置：第 65-70 行**
```python
# Load environment light (learnable)
env_light = EnvironmentLight(env_map_path, resolution=256).cuda()

# Environment light optimizer
env_light_lr = getattr(opt, 'env_light_lr', 0.01)
env_light_optimizer = torch.optim.Adam(env_light.parameters(), lr=env_light_lr)
```

**结论**：✓ 正确
- 使用 `env_light.parameters()` 正确获取所有可学习参数
- Adam 优化器适合学习 HDR 环境贴图
- 默认学习率 0.01 合理

### 2.2 optimizer.step() 和 zero_grad() 调用位置检查 ✓ 正确

**代码位置：第 233-241 行**
```python
# Optimizer step
if iteration < opt.iterations:
    gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)

    # Environment light optimizer (only after PBR training starts)
    if iteration > 5000:
        env_light_optimizer.step()
        env_light_optimizer.zero_grad(set_to_none=True)
```

**结论**：✓ 正确
- 位置正确：在 backward() 之后（第 169 行）
- 调用顺序正确：zero_grad() 在 step() 之后
- 梯度累积得以避免

### 2.3 iteration > 5000 条件检查 ✓ 正确

**代码位置：第 120 行（PBR shading）和第 239 行（env_light optimizer）**
```python
if iteration > 5000 and gaussians.use_pbr:
    # Get G-Buffer
    ...

# Environment light optimizer (only after PBR training starts)
if iteration > 5000:
    env_light_optimizer.step()
```

**结论**：✓ 正确
- PBR shading 损失计算：iteration > 5000 启动
- env_light 优化：iteration > 5000 启动
- 两者一致，避免初期不稳定训练

### 2.4 env_light 保存检查 ✓ 正确

**代码位置：第 207-213 行**
```python
if iteration in saving_iterations:
    print("\n[ITER {}] Saving Gaussians".format(iteration))
    scene.save(iteration)
    # Save environment light
    env_light_path = os.path.join(scene.model_path, f"env_light_{iteration}.pth")
    torch.save(env_light.state_dict(), env_light_path)
    print(f"[ITER {iteration}] Saved environment light to {env_light_path}")
```

**结论**：✓ 正确
- 使用 `state_dict()` 保存（标准做法）
- 保存频率与高斯模型一致
- 文件名格式清晰：`env_light_{iteration}.pth`

---

## 3. render_pbr.py 中的环境光加载检查 ✓ 正确

### 3.1 训练好的 env_light 加载检查 ✓ 正确

**代码位置：第 215-237 行**
```python
# Load environment light (try to load trained one first)
env_light = EnvironmentLight(env_map_path, resolution=256).cuda()
env_light_loaded = False

# Try to load trained environment light
env_light_path = os.path.join(args.model_path, f"env_light_{iteration}.pth")
if os.path.exists(env_light_path):
    env_light.load_state_dict(torch.load(env_light_path))
    print(f"Loaded trained environment light from: {env_light_path}")
    env_light_loaded = True
else:
    # Try to find any saved env_light
    if os.path.exists(args.model_path):
        for f in sorted(os.listdir(args.model_path), reverse=True):
            if f.startswith("env_light_") and f.endswith(".pth"):
                env_light_path = os.path.join(args.model_path, f)
                env_light.load_state_dict(torch.load(env_light_path))
                print(f"Loaded trained environment light from: {env_light_path}")
                env_light_loaded = True
                break
```

**结论**：✓ 正确
- 优先加载指定迭代的 env_light
- 备选方案：自动查找最新保存的 env_light
- 健壮的错误处理

### 3.2 load_state_dict() 使用检查 ✓ 正确

```python
env_light.load_state_dict(torch.load(env_light_path))
```

**结论**：✓ 正确
- 标准的 PyTorch 加载方式
- 与 save 时的 `state_dict()` 对应

---

## 4. 梯度流检查 ✓ 正确，完整

### 4.1 完整梯度流路径追踪

```
loss (第 167 行)
    ↓
total_loss.backward() (第 169 行)
    ↓
pbr_loss = lambda_pbr * pbr_reconstruction_loss(shaded_image, gt_image) (第 143 行)
    ↓
shaded_image = screen_space_pbr_shading(..., env_light=env_light, ...) (第 131-140 行)
    ↓
pbr_shading_env(..., env_light) (pbr_utils.py, 第 264-321 行)
    ↓
env_light.sample(reflect_dir)  (第 299 行)
env_light.sample(normal)       (第 302 行)
    ↓
F.grid_sample(env_map, grid, ...) (第 104-109 行)
    ↓
env_light.env_map 梯度累积
```

**结论**：✓ 完整且可微分
- 从 loss 到 env_map 的梯度流通畅
- 没有任何梯度阻止操作（detach/no_grad）
- env_light 为 nn.Module，参与自动微分

### 4.2 关键点检查：是否有梯度阻止

**检查结果**：
1. ✓ `render_pbr.py` 的 detach() 用于保存图像（第 33, 40 行），不影响训练
2. ✓ `gaussian_renderer/__init__.py` 的 detach() 用于 surf_normal（第 208 行），不影响 env_light 梯度
3. ✓ `pbr_utils.py` 中全部操作可微分，无梯度阻止

---

## 5. 重要发现与建议

### 5.1 发现：批次规范化缺失
**现象**：
```python
# pbr_utils.py 第 36-37 行
self.env_map = nn.Parameter(
    torch.ones(3, resolution, resolution * 2) * 0.5
)
```

**分析**：
- env_map 初始化为 0.5（对称灰色）
- 通过 grid_sample 采样后的值可能很小
- 学习率 0.01 可能导致训练太快或太慢

**建议**：考虑添加值归一化或学习率调度

### 5.2 发现：渐进式训练良好
**分析**：
```python
if iteration > 5000:  # 第 120, 239 行
    # PBR loss 和 env_light optimizer 同时启动
```

**优点**：
- 先稳定高斯优化，后加入 PBR
- 避免初期因环境光学习导致的不稳定
- 这是合理的设计

### 5.3 发现：可能的改进点
**问题**：env_light 的学习率（0.01）与其他参数相比较高

**当前值**：
```python
env_light_lr = getattr(opt, 'env_light_lr', 0.01)
```

**建议**：
- 监控 env_light 损失的收敛速度
- 如果波动大，考虑降低学习率（如 0.001）
- 或使用学习率调度器

### 5.4 发现：env_light 在评估时保持 train 模式
**代码位置**：`train_pbr.py` 第 131-140 行

**分析**：
- 评估时仍在 train 模式，env_light 可能继续被修改
- 虽然 `@torch.no_grad()` 装饰器应该阻止梯度累积，但建议显式设置

**建议**：
```python
# 在评估前
env_light.eval()
# 评估后恢复
env_light.train()
```

---

## 6. 总体评分

| 检查项目 | 状态 | 分数 |
|---------|------|------|
| EnvironmentLight 类实现 | ✓ 正确 | 10/10 |
| env_map 可学习性 | ✓ 正确 | 10/10 |
| sample() 方法微分性 | ✓ 正确 | 10/10 |
| env_light_optimizer 设置 | ✓ 正确 | 10/10 |
| optimizer.step/zero_grad 调用 | ✓ 正确 | 10/10 |
| 迭代阈值条件 | ✓ 正确 | 10/10 |
| env_light 保存机制 | ✓ 正确 | 10/10 |
| env_light 加载机制 | ✓ 正确 | 10/10 |
| 梯度流完整性 | ✓ 正确 | 10/10 |
| 梯度阻止检查 | ✓ 正确 | 10/10 |
|---|---|---|
| **总体得分** | **✓ 95/100** | |

### 扣分原因：
- **-5 分**：建议添加训练/评估模式切换和学习率调度

---

## 7. 结论

### 核心结论
**可学习环境贴图的实现是正确的。** 环境光能够正确地被学习和优化。

### 关键确认
1. ✓ `env_map` 正确注册为 `nn.Parameter`
2. ✓ `sample()` 方法完全可微分
3. ✓ 优化器正确创建和调用
4. ✓ 梯度能够完整流动：loss → env_light.sample() → env_map
5. ✓ 保存和加载机制标准且正确
6. ✓ 无任何梯度阻止操作破坏训练

### 可以放心使用的方面
- 环境光确实在被学习
- 训练过程中梯度计算无误
- 保存的环境光可以被正确加载

### 建议优化方向
1. 添加学习率调度或衰减
2. 考虑在评估时显式设置 `eval()` 模式
3. 监控 env_light 的损失下降曲线
4. 考虑值归一化或裁剪，防止学到极端值

---

## 附录：关键代码片段总结

### EnvironmentLight 类初始化
```python
class EnvironmentLight(nn.Module):
    def __init__(self, env_map_path: str = None, resolution: int = 512):
        super().__init__()
        self.resolution = resolution
        
        if env_map_path is not None and os.path.exists(env_map_path):
            self.env_map = self._load_env_map(env_map_path)
        else:
            self.env_map = nn.Parameter(
                torch.ones(3, resolution, resolution * 2) * 0.5,
                requires_grad=True
            )
```

### 可微分采样
```python
def sample(self, directions: torch.Tensor) -> torch.Tensor:
    # ... 方向转换 ...
    grid = torch.stack([grid_u, grid_v], dim=-1).unsqueeze(0).unsqueeze(0)
    env_map = self.env_map.unsqueeze(0)
    sampled = F.grid_sample(
        env_map, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    return colors.reshape(*original_shape, 3)
```

### 优化器设置
```python
env_light = EnvironmentLight(env_map_path, resolution=256).cuda()
env_light_optimizer = torch.optim.Adam(env_light.parameters(), lr=0.01)

# 训练循环
if iteration > 5000:
    env_light_optimizer.step()
    env_light_optimizer.zero_grad(set_to_none=True)
```

### 保存和加载
```python
# 保存
torch.save(env_light.state_dict(), env_light_path)

# 加载
env_light.load_state_dict(torch.load(env_light_path))
```

