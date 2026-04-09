#!/bin/bash
#
# PhysicalFish Hackathon 预检脚本
# 运行这个确保一切就绪
#

set -e

echo "🔍 PhysicalFish Hackathon Pre-Flight Check"
echo "=========================================="
echo ""

# 检查 Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✅ Python: $PYTHON_VERSION"
else
    echo "❌ Python3 not found"
    exit 1
fi

# 检查 pip 包
echo ""
echo "📦 Checking Python packages..."
python3 -c "import numpy" 2>/dev/null && echo "✅ numpy" || echo "❌ numpy (pip install numpy)"

# 检查项目结构
echo ""
echo "📁 Checking project structure..."
FILES=(
    "fast_demo.py"
    "real-data/capture_real.py"
    "verification/verify_physics.py"
    "autoresearch/optimize_loop.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file missing"
    fi
done

# 测试 fast demo
echo ""
echo "🧪 Testing fast demo (3 iterations)..."
cd ~/Downloads/physicalfish-demo
python3 fast_demo.py --target 0.85 --iterations 3 --no-interactive > /tmp/demo_test.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Fast demo runs successfully"
    grep "DEMO COMPLETE" /tmp/demo_test.log | head -1
else
    echo "❌ Fast demo failed"
    cat /tmp/demo_test.log
fi

# 检查 Godot（可选）
echo ""
echo "🎮 Checking Godot (optional)..."
if command -v godot &> /dev/null; then
    GODOT_VERSION=$(godot --version 2>&1 | head -1)
    echo "✅ Godot: $GODOT_VERSION"
else
    echo "⚠️  Godot not found (optional, fast_demo.py works without it)"
fi

# 生成演示数据
echo ""
echo "📊 Generating sample results..."
python3 fast_demo.py --target 0.90 --iterations 12 > /tmp/sample_run.log 2>&1
if [ $? -eq 0 ]; then
    FINAL_SCORE=$(grep "Final Score:" /tmp/sample_run.log | awk '{print $3}')
    echo "✅ Sample run complete: Final score $FINAL_SCORE"
fi

echo ""
echo "=========================================="
echo "🎯 Pre-flight check complete!"
echo ""
echo "Tomorrow's checklist:"
echo "  [ ] Laptop + charger"
echo "  [ ] This repo on Desktop"
echo "  [ ] fast_demo.py tested"
echo "  [ ] Business cards (optional)"
echo ""
echo "Run demo: python3 fast_demo.py"
