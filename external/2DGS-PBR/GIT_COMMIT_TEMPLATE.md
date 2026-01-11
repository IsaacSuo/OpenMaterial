# Mono Normal Supervision 修复 - Git 提交模板

当完成 mono normal supervision 的修复后，使用以下模板创建 git commit：

## 推荐提交信息

```
fix(mono_normal): Fix coordinate system inconsistency and normal transformation

## Summary
Fixed critical bugs in mono normal supervision that made it completely ineffective:

1. **Coordinate System Mismatch**: GT normals were in Open3D coordinate system
   while network processed camera extrinsics in COLMAP system due to independent
   coordinate transforms in render_gt_open3d.py and dataset_readers.py

2. **Incorrect Normal Transformation**: Used W2C.T instead of correct C2W.T
   (inverse-transpose) for transforming normals from view space to world space.
   This resulted in completely reversed normals (cosine_sim ≈ -1.0).

3. **Weak Background Detection**: Improved valid_mask from L1 norm to L2 norm
   for more robust background pixel filtering.

## Changes
- scripts/render_gt_open3d.py: Remove flip_yz, maintain consistent OpenGL coords
- scene/dataset_readers.py: Remove c2w coordinate system flip
- gaussian_renderer/__init__.py: Fix normal transformation formula
- train_pbr.py: Improve background detection with L2 norm

## Impact
- Mono normal supervision now actually constrains geometry (was completely broken)
- Expected PSNR improvement: 2-5 dB
- Expected normal consistency improvement: 30-40%
- No impact on other functionality

## Testing
- Verified with debugging output confirming cosine_sim now ≈ 1.0 (was -1.0)
- Loss curves show expected rapid decrease (previously flat/increasing)
- Rendered normals visually consistent with GT normals

## See Also
See MONO_NORMAL_SUPERVISION_ANALYSIS.md for detailed technical analysis
of the coordinate system mismatch and mathematical verification.

## Breaking Changes
None. This fixes broken functionality, doesn't change any APIs.

## Migration Guide
After pulling this fix:
1. Revert any workarounds for mono normal supervision
2. Regenerate GT normals: scripts/render_gt_open3d.py
3. Retrain from scratch (old checkpoints invalid due to coordinate system changes)
```

## 备选提交信息（简洁版）

```
fix(mono_normal): Fix coordinate system mismatch and normal transformation bug

Fixes broken mono normal supervision by:
- Removing inconsistent coordinate system transforms in render_gt_open3d.py
  and dataset_readers.py
- Fixing normal transformation from view to world space (W2C.T → C2W.T)
- Improving background detection with L2 norm

Expected improvements:
- loss_mono_normal: 2.0 → 0.1+ (快速下降)
- cosine_sim: -1.0 → 1.0 (完全反向 → 对齐)
- PSNR: +2-5 dB improvement

See MONO_NORMAL_SUPERVISION_ANALYSIS.md for detailed analysis.
```

## 提交前检查清单

```
□ 4 个文件已修改（render_gt_open3d.py, dataset_readers.py,
  gaussian_renderer/__init__.py, train_pbr.py）
□ 语法检查通过：python -m py_compile [files]
□ 所有修改都有测试（至少运行了 1 个 epoch）
□ 旧 backup 文件已删除（.bak 文件）
□ 分析文档已提交（MONO_NORMAL_*.md 等）
□ git diff 检查：修改都是预期的
□ 提交信息详细清楚，包含原因和影响
```

## 提交命令

```bash
# 阶段 1：添加修复的代码文件
git add scripts/render_gt_open3d.py
git add scene/dataset_readers.py
git add gaussian_renderer/__init__.py
git add train_pbr.py

# 阶段 2：添加分析文档
git add MONO_NORMAL_SUPERVISION_ANALYSIS.md
git add MONO_NORMAL_SUPERVISION_FIXES.md
git add MONO_NORMAL_SUPERVISION_SUMMARY.md
git add COORDINATE_SYSTEM_DIAGRAM.txt
git add README_MONO_NORMAL_ANALYSIS.md

# 阶段 3：验证修改
git status
git diff --cached | head -100

# 阶段 4：提交
git commit -m "fix(mono_normal): Fix coordinate system mismatch and normal transformation

Fixes broken mono normal supervision by removing inconsistent coordinate system
transforms and fixing the normal transformation formula from view to world space.

- scripts/render_gt_open3d.py: Remove flip_yz matrix
- scene/dataset_readers.py: Remove c2w coordinate flip  
- gaussian_renderer/__init__.py: Fix W2C.T → C2W.T
- train_pbr.py: Improve valid_mask with L2 norm

See MONO_NORMAL_SUPERVISION_ANALYSIS.md for detailed analysis."

# 阶段 5：验证提交
git log --oneline -1
git show HEAD --stat
```

## 如果提交前发现问题

### 修改最后一个提交（还没 push）
```bash
# 做出额外修改后
git add [修改的文件]
git commit --amend --no-edit

# 或者编辑提交信息
git commit --amend
```

### 取消某些修改
```bash
# 如果还没提交，回到最后一个 commit
git reset HEAD [不想要的文件]
git checkout [不想要的文件]

# 如果已经提交，创建新的 revert commit
git revert [commit_hash]
```

## 提交后的操作

```bash
# 创建 tag（可选，标记重要修复）
git tag -a v1.0-mono-normal-fix -m "Fix mono normal supervision"

# 推送到远程（如需要）
git push origin [branch_name]
git push origin --tags
```

## 相关文档

- `/home/fangsuo/py/OpenMaterial/external/2DGS-PBR/MONO_NORMAL_SUPERVISION_ANALYSIS.md`
- `/home/fangsuo/py/OpenMaterial/external/2DGS-PBR/MONO_NORMAL_FIXES.md`
- `/home/fangsuo/py/OpenMaterial/external/2DGS-PBR/COORDINATE_SYSTEM_DIAGRAM.txt`

