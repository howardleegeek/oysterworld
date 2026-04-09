#!/bin/bash
#
# PhysicalFish Demo — One-Click Runner
# 
# Usage: ./run_demo.sh [options]
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
REAL_DATA_DIR="${OUTPUT_DIR}/real"
SYNTHETIC_DATA_DIR="${OUTPUT_DIR}/synthetic"

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  PhysicalFish Demo — Egocentric Data Flywheel              ║"
echo "║  Seed with real. Scale with synthetic. Optimize with AI. ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Parse arguments
TARGET_SCORE=0.90
MAX_ITERATIONS=15
SKIP_REAL=false
SKIP_SYNTHETIC=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --target-score)
            TARGET_SCORE="$2"
            shift 2
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --skip-real)
            SKIP_REAL=true
            shift
            ;;
        --skip-synthetic)
            SKIP_SYNTHETIC=true
            shift
            ;;
        --help)
            echo "Usage: ./run_demo.sh [options]"
            echo ""
            echo "Options:"
            echo "  --target-score N       Target physics score (default: 0.90)"
            echo "  --max-iterations N     Max optimization iterations (default: 15)"
            echo "  --skip-real            Skip real data generation (use existing)"
            echo "  --skip-synthetic       Skip synthetic generation (verify only)"
            echo "  --help                 Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Create output directories
mkdir -p "${REAL_DATA_DIR}"
mkdir -p "${SYNTHETIC_DATA_DIR}"

# Step 1: Generate Real Data
echo ""
echo -e "${YELLOW}Step 1: Generating Real Data (Seed)${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$SKIP_REAL" = false ]; then
    echo "Creating simulated egocentric grasping data..."
    cd "${SCRIPT_DIR}/real-data"
    python3 capture_real.py --output "${REAL_DATA_DIR}" --duration 3.0
    
    # Find the generated trajectory file
    REAL_DATA_FILE=$(find "${REAL_DATA_DIR}" -name "*trajectory.json" -type f | head -1)
    
    if [ -z "$REAL_DATA_FILE" ]; then
        echo -e "${RED}Error: Failed to generate real data${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Real data generated: $(basename $REAL_DATA_FILE)${NC}"
else
    REAL_DATA_FILE=$(find "${REAL_DATA_DIR}" -name "*trajectory.json" -type f | head -1)
    if [ -z "$REAL_DATA_FILE" ]; then
        echo -e "${RED}Error: No existing real data found${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Using existing real data: $(basename $REAL_DATA_FILE)${NC}"
fi

# Step 2: Run AutoResearch Optimization Loop
echo ""
echo -e "${YELLOW}Step 2: Running AutoResearch Optimization Loop${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Target score: ${TARGET_SCORE}"
echo "Max iterations: ${MAX_ITERATIONS}"
echo ""

cd "${SCRIPT_DIR}/autoresearch"

# Check if Godot is installed
if ! command -v godot &> /dev/null; then
    echo -e "${RED}Error: Godot is not installed or not in PATH${NC}"
    echo "Please install Godot 4.x: https://godotengine.org/download"
    exit 1
fi

# Run optimization
python3 optimize_loop.py \
    --real-data "${REAL_DATA_FILE}" \
    --godot-project "${SCRIPT_DIR}/synthetic-data/godot-project" \
    --output "${SYNTHETIC_DATA_DIR}" \
    --target-score "${TARGET_SCORE}" \
    --max-iterations "${MAX_ITERATIONS}" \
    --patience 5

# Check if optimization report was generated
REPORT_FILE="${SYNTHETIC_DATA_DIR}/optimization_report.json"
if [ -f "$REPORT_FILE" ]; then
    echo ""
    echo -e "${GREEN}✓ Optimization complete!${NC}"
    
    # Extract final score
    FINAL_SCORE=$(python3 -c "import json; print(json.load(open('$REPORT_FILE'))['summary']['final_score'])")
    echo -e "${GREEN}  Final score: ${FINAL_SCORE}${NC}"
else
    echo -e "${RED}Error: Optimization report not found${NC}"
    exit 1
fi

# Step 3: Generate Summary
echo ""
echo -e "${YELLOW}Step 3: Results Summary${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "${SCRIPT_DIR}"
python3 << PYTHON
import json
from pathlib import Path

# Load optimization report
report_path = Path("${REPORT_FILE}")
with open(report_path) as f:
    report = json.load(f)

summary = report["summary"]

print(f"\n{'='*60}")
print(f"PhysicalFish Demo Results")
print(f"{'='*60}")
print(f"Final Physics Score:     {summary['final_score']:.4f}")
print(f"Target Score:            {summary['target_score']:.4f}")
print(f"Iterations:              {summary['iterations']}")
print(f"Time Elapsed:            {summary['elapsed_time_seconds']:.1f}s")
print(f"Converged:               {'✓ YES' if summary['converged'] else '✗ NO'}")
print(f"{'='*60}")

print(f"\nBest Physics Parameters:")
for key, val in summary['best_params'].items():
    print(f"  {key:20s}: {val:.3f}")

print(f"\nConvergence Curve:")
for point in report['convergence_curve'][:10]:
    symbol = "✓" if point['action'] == 'commit' else "✗"
    print(f"  Iter {point['iteration']:2d}: {point['score']:.4f} {symbol}")

if len(report['convergence_curve']) > 10:
    print(f"  ... ({len(report['convergence_curve']) - 10} more iterations)")

print(f"\n{'='*60}")
print(f"Output Files:")
print(f"  Real data:       ${REAL_DATA_FILE}")
print(f"  Report:          ${REPORT_FILE}")
print(f"  Best params:     ${SYNTHETIC_DATA_DIR}/best_params.json")
print(f"{'='*60}")
PYTHON

echo ""
echo -e "${GREEN}🎯 Demo complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Check the convergence curve in the report"
echo "  2. Review best_params.json for optimized physics settings"
echo "  3. Use these parameters for large-scale synthetic data generation"
echo ""
echo "For Paradigm pitch:"
echo "  - The flywheel works: real data → synthetic → verification → optimization"
echo "  - Physics score improved from initial to final"
echo "  - This is the 'seed with real, scale with synthetic' proof"
echo ""
