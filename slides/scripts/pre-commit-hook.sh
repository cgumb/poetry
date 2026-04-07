#!/usr/bin/env bash
#
# Pre-commit hook to ensure slides/presentation.pdf is up-to-date
#
# If slides/presentation.md is staged, rebuild the PDF and stage it.
#

set -e

# Check if presentation.md is being committed
if git diff --cached --name-only | grep -q "^slides/presentation.md$"; then
    echo "======================================================================"
    echo "SLIDES CHANGED: Rebuilding presentation.pdf..."
    echo "======================================================================"
    echo ""

    # Save current directory
    REPO_ROOT=$(git rev-parse --show-toplevel)

    # Check if pandoc is available
    if ! command -v pandoc &> /dev/null; then
        echo "ERROR: pandoc not found!"
        echo "Please install pandoc: conda install -c conda-forge pandoc"
        echo ""
        echo "Or skip this check with: git commit --no-verify"
        exit 1
    fi

    # Build slides
    cd "$REPO_ROOT/slides"

    if bash build_slides.sh > /tmp/slides_build.log 2>&1; then
        echo "✓ Slides rebuilt successfully"
        echo ""

        # Stage the updated PDF
        git add presentation.pdf
        echo "✓ presentation.pdf auto-staged"
        echo ""
        echo "======================================================================"
        echo "READY TO COMMIT"
        echo "======================================================================"
    else
        echo "ERROR: Failed to build slides!"
        echo ""
        echo "Build log:"
        cat /tmp/slides_build.log
        echo ""
        echo "Fix the error or skip this check with: git commit --no-verify"
        exit 1
    fi
fi

exit 0
