#!/bin/bash
#
# Build presentation slides from markdown to PDF using pandoc + beamer
#

set -e

SLIDES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="$SLIDES_DIR/presentation.md"
OUTPUT="$SLIDES_DIR/presentation.pdf"

echo "=========================================="
echo "Building Presentation Slides"
echo "=========================================="
echo ""
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo ""

# Check pandoc is available
if ! command -v pandoc &> /dev/null; then
    echo "ERROR: pandoc not found!"
    echo "Install with: conda install -c conda-forge pandoc"
    exit 1
fi

# Build PDF with beamer
echo "Running pandoc..."
pandoc "$INPUT" \
    -t beamer \
    --pdf-engine=pdflatex \
    --slide-level=1 \
    -V theme:Madrid \
    -V colortheme:default \
    -V fontsize:10pt \
    -V aspectratio:169 \
    -o "$OUTPUT"

echo ""
echo "✓ Slides built successfully!"
echo "  Output: $OUTPUT"
echo ""
echo "To view: evince $OUTPUT"
echo "=========================================="
