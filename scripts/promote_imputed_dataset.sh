#!/bin/bash
#
# Promote the imputed dataset to be the canonical poems.parquet
# and prepare shared data for students
#

set -e

SHARED_DATA_DIR="/shared/courseSharedFolders/161588outer/161588/poetry_data"

echo "========================================================================"
echo "PROMOTING IMPUTED DATASET TO CANONICAL"
echo "========================================================================"
echo ""

# Check that imputed dataset exists
if [ ! -f "data/poems_imputed.parquet" ]; then
    echo "ERROR: data/poems_imputed.parquet not found!"
    echo "Run the LLM imputation pipeline first."
    exit 1
fi

# Backup original if it exists and hasn't been backed up yet
if [ -f "data/poems.parquet" ]; then
    if [ ! -f "data/poems.parquet.original" ]; then
        echo "Backing up original poems.parquet -> poems.parquet.original"
        cp data/poems.parquet data/poems.parquet.original
    else
        echo "Original backup already exists: data/poems.parquet.original"
    fi
fi

# Replace with imputed version
echo "Replacing data/poems.parquet with imputed version..."
cp data/poems_imputed.parquet data/poems.parquet
echo "✓ data/poems.parquet now contains imputed poet names"
echo ""

echo "========================================================================"
echo "PREPARING SHARED DATA FOR STUDENTS"
echo "========================================================================"
echo ""

# Create shared directory
mkdir -p "$SHARED_DATA_DIR"
echo "Shared data directory: $SHARED_DATA_DIR"
echo ""

# Copy preprocessed files to shared location
echo "Copying preprocessed files to shared location..."
echo ""

FILES_TO_SHARE=(
    "data/poems.parquet:Core dataset with imputed poets"
    "data/embeddings.npy:384-dim poem embeddings"
    "data/proj2d.npy:2D projection of all poems"
    "data/proj2d_reducer.pkl:UMAP reducer for 2D projection"
    "data/poet_centroids.parquet:Poet metadata (name, poem count)"
    "data/poet_centroids.npy:384-dim poet centroid embeddings"
    "data/poet_centroids_2d.npy:2D projection of poet centroids"
)

for entry in "${FILES_TO_SHARE[@]}"; do
    IFS=':' read -r filepath description <<< "$entry"
    filename=$(basename "$filepath")

    if [ -f "$filepath" ]; then
        echo "  Copying $filename ... ($description)"
        cp "$filepath" "$SHARED_DATA_DIR/$filename"
    else
        echo "  WARNING: $filepath not found - skipping"
    fi
done

echo ""
echo "Setting permissions for student read access..."
chmod -R a+rX "$SHARED_DATA_DIR"
echo ""

echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"
echo "Local data:"
echo "  ✓ data/poems.parquet <- imputed version (original backed up)"
echo ""
echo "Shared data for students:"
echo "  ✓ $SHARED_DATA_DIR"
ls -lh "$SHARED_DATA_DIR" 2>/dev/null || echo "  (directory empty or doesn't exist)"
echo ""
echo "Students can now run: bash scripts/setup_shared_data.sh"
echo "========================================================================"
