#!/bin/bash
# 自动应用坐标系修复

set -e

FILE="scripts/render_gt_open3d.py"
BACKUP="scripts/render_gt_open3d.py.backup"
LINENUM=163

echo "2DGS-PBR Coordinate System Fix"
echo "=============================="
echo ""

if [ ! -f "$FILE" ]; then
    echo "Error: $FILE not found!"
    exit 1
fi

# Create backup
if [ ! -f "$BACKUP" ]; then
    cp "$FILE" "$BACKUP"
    echo "[INFO] Backup created: $BACKUP"
else
    echo "[INFO] Backup already exists: $BACKUP"
fi

# Check if already patched
if grep -q "flip_yz_3x3" "$FILE"; then
    echo "[WARN] File appears to be already patched!"
    echo "Check $FILE manually if needed."
    exit 0
fi

echo "[INFO] Applying fix to $FILE at line $LINENUM..."

# Use Python to apply the patch more reliably
python3 << 'PYTHON_PATCH'
import sys

file_path = "scripts/render_gt_open3d.py"

with open(file_path, 'r') as f:
    lines = f.readlines()

# Find the line "normals = np.asarray(mesh_normal.vertex_normals)"
insert_line = None
for i, line in enumerate(lines):
    if 'normals = np.asarray(mesh_normal.vertex_normals)' in line:
        insert_line = i + 1
        break

if insert_line is None:
    print("[ERROR] Could not find insertion point in", file_path)
    sys.exit(1)

# Prepare the fix code
fix_code = '''
    # CRITICAL FIX: Transform normals to match flipped camera coordinate system
    # The camera is transformed with flip_yz, so normals must be transformed too
    flip_yz_3x3 = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float32)
    normals = normals @ flip_yz_3x3.T  # Apply rotation to row vectors
    
'''

# Find indentation of the line before
indent = len(lines[insert_line - 1]) - len(lines[insert_line - 1].lstrip())
fix_code_indented = '\n'.join([' ' * indent + line if line.strip() else line 
                               for line in fix_code.split('\n')])

# Insert the fix
lines.insert(insert_line, fix_code_indented)

# Write back
with open(file_path, 'w') as f:
    f.writelines(lines)

print(f"[SUCCESS] Patch applied! Fix inserted at line {insert_line + 1}")

PYTHON_PATCH

if [ $? -eq 0 ]; then
    echo ""
    echo "[SUCCESS] Fix applied successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Review the changes:"
    echo "   diff -u $BACKUP $FILE"
    echo ""
    echo "2. Regenerate GT data:"
    echo "   python scripts/render_gt_open3d.py -s <dataset_path> --debug"
    echo ""
    echo "3. Retrain:"
    echo "   python train_pbr.py -s <dataset_path> ..."
    echo ""
else
    echo "[ERROR] Failed to apply patch!"
    echo "Restoring from backup..."
    cp "$BACKUP" "$FILE"
    exit 1
fi
