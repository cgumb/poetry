#!/bin/bash
#
# Update figure paths in presentation.md with most recent benchmark results
#

set -e

SLIDES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESENTATION="$SLIDES_DIR/presentation.md"
RESULTS_DIR="$SLIDES_DIR/../results"

echo "=========================================="
echo "Updating Figure Paths"
echo "=========================================="
echo ""

# Find most recent pedagogy benchmark directory
LATEST_PEDAGOGY=$(ls -td "$RESULTS_DIR"/pedagogy_* 2>/dev/null | head -1)

if [ -z "$LATEST_PEDAGOGY" ]; then
    echo "ERROR: No pedagogy benchmark results found!"
    echo ""
    echo "Run the pedagogical benchmarks first:"
    echo "  sbatch scripts/pedagogical_benchmarks.slurm"
    echo ""
    echo "Or generate figures manually:"
    echo "  python scripts/visualize_scaling.py results/*.csv --output-dir results/pedagogy_manual/"
    exit 1
fi

PEDAGOGY_DIR=$(basename "$LATEST_PEDAGOGY")

echo "Found latest results: $PEDAGOGY_DIR"
echo ""

# Update all pedagogy_* paths in presentation.md
echo "Updating $PRESENTATION..."
sed -i "s|pedagogy_\*|$PEDAGOGY_DIR|g" "$PRESENTATION"

echo "✓ Updated figure paths to: ../results/$PEDAGOGY_DIR/"
echo ""

# List available figures
echo "Available figures:"
ls -1 "$LATEST_PEDAGOGY"/*.png 2>/dev/null | while read fig; do
    echo "  - $(basename "$fig")"
done

echo ""
echo "=========================================="
echo "Ready to build slides:"
echo "  bash build_slides.sh"
echo "=========================================="
