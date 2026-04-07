#!/bin/bash
#
# Setup script for students: Symlink preprocessed data from shared location
#
# This avoids having students regenerate embeddings (which takes hours)
# and lets them jump straight into the GP active learning work.
#

set -e

SHARED_DATA_DIR="/shared/courseSharedFolders/161588outer/161588/poetry_data"
LOCAL_DATA_DIR="data"

echo "========================================================================"
echo "POETRY GP: SHARED DATA SETUP"
echo "========================================================================"
echo ""
echo "This script creates symlinks to preprocessed data files so you don't"
echo "need to regenerate embeddings (which takes ~2 hours for 85k poems)."
echo ""

# Check that shared data exists
if [ ! -d "$SHARED_DATA_DIR" ]; then
    echo "ERROR: Shared data directory not found!"
    echo "Expected: $SHARED_DATA_DIR"
    echo ""
    echo "Contact the instructor if this directory is missing."
    exit 1
fi

# Create local data directory if it doesn't exist
mkdir -p "$LOCAL_DATA_DIR"

# Files to symlink from shared location
SHARED_FILES=(
    "poems.parquet"
    "embeddings.npy"
    "proj2d.npy"
    "proj2d_reducer.pkl"
    "poet_centroids.parquet"
    "poet_centroids.npy"
    "poet_centroids_2d.npy"
)

echo "Creating symlinks to shared data files..."
echo ""

for filename in "${SHARED_FILES[@]}"; do
    shared_file="$SHARED_DATA_DIR/$filename"
    local_file="$LOCAL_DATA_DIR/$filename"

    if [ ! -f "$shared_file" ]; then
        echo "  WARNING: $shared_file not found in shared location - skipping"
        continue
    fi

    # If local file exists and is NOT a symlink, back it up
    if [ -f "$local_file" ] && [ ! -L "$local_file" ]; then
        backup="$local_file.backup.$(date +%Y%m%d_%H%M%S)"
        echo "  Backing up existing $filename -> $(basename $backup)"
        mv "$local_file" "$backup"
    fi

    # Remove existing symlink if it exists
    if [ -L "$local_file" ]; then
        rm "$local_file"
    fi

    # Create new symlink
    ln -s "$shared_file" "$local_file"
    echo "  ✓ $filename -> $shared_file"
done

echo ""
echo "========================================================================"
echo "SETUP COMPLETE"
echo "========================================================================"
echo ""
echo "Symlinked data files:"
ls -lh "$LOCAL_DATA_DIR" | grep -- '->'
echo ""
echo "You can now run the application or benchmarks:"
echo "  python scripts/app/run_gp_session.py"
echo "  python scripts/app/render_session_plots.py"
echo "  sbatch scripts/pedagogical_benchmarks.slurm"
echo ""
echo "Note: These are symlinks to READ-ONLY shared data."
echo "Any session files you create will be written to your own results/ directory."
echo "========================================================================"
