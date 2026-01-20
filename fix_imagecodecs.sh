#!/bin/bash
# 修复 imagecodecs JPEG-XL 支持的脚本

set -e

echo "============================================================"
echo "Fixing imagecodecs JPEG-XL support"
echo "============================================================"

# 检查是否在conda环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Warning: Not in a conda environment. Please activate your conda environment first."
    exit 1
fi

echo "Current conda environment: $CONDA_DEFAULT_ENV"

# 步骤1: 确保 libjxl 已安装
echo ""
echo "Step 1: Ensuring libjxl is installed..."
conda install -c conda-forge libjxl -y

# 步骤2: 卸载 conda 的 imagecodecs（如果存在）
echo ""
echo "Step 2: Uninstalling conda imagecodecs..."
conda remove imagecodecs -y 2>/dev/null || echo "  (imagecodecs not installed via conda)"

# 步骤3: 卸载 pip 的 imagecodecs（如果存在）
echo ""
echo "Step 3: Uninstalling pip imagecodecs..."
pip uninstall -y imagecodecs 2>/dev/null || echo "  (imagecodecs not installed via pip)"

# 步骤4: 设置环境变量以帮助找到 libjxl
echo ""
echo "Step 4: Setting up environment for pip installation..."

# 获取 conda 环境的路径
CONDA_PREFIX="${CONDA_PREFIX:-$CONDA_DEFAULT_ENV}"

# 查找 libjxl 库路径
if [ -d "$CONDA_PREFIX/lib" ]; then
    export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"
    export CPATH="$CONDA_PREFIX/include:$CPATH"
    echo "  Set PKG_CONFIG_PATH=$PKG_CONFIG_PATH"
    echo "  Set LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "  Set LIBRARY_PATH=$LIBRARY_PATH"
    echo "  Set CPATH=$CPATH"
fi

# 步骤5: 使用 pip 安装 imagecodecs（会从源码编译并检测 libjxl）
echo ""
echo "Step 5: Installing imagecodecs via pip (will compile with JPEG-XL support)..."
pip install --no-binary imagecodecs imagecodecs

# 步骤6: 验证安装
echo ""
echo "Step 6: Verifying installation..."
python -c "
import imagecodecs
print(f'imagecodecs version: {imagecodecs.__version__}')
jpegxl = imagecodecs.JPEGXL
is_stub = 'STUB' in str(jpegxl) or 'STUB' in str(type(jpegxl))
print(f'JPEGXL available: {jpegxl}')
print(f'Is STUB: {is_stub}')
if is_stub:
    print('✗ JPEG-XL support is still not available')
    exit(1)
else:
    print('✓ JPEG-XL support is available!')
"

echo ""
echo "============================================================"
echo "Installation complete! Please test with: python check_imagecodecs.py"
echo "============================================================"
