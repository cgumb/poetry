#!/bin/bash
# Test OLD ScaLAPACK path (pre-built matrix, not features)
# This uses the legacy centralized path, not Milestone 1B

set -euo pipefail

# Create test data
python3 << 'EOF'
import numpy as np
import json
from pathlib import Path

# Test problem
np.random.seed(42)
n = 100
d = 384

# Generate features
x = np.random.randn(n, d).astype(np.float64)
x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12

# Generate targets
y = np.random.randn(n).astype(np.float64)

# Build kernel matrix in Python (this is the OLD way)
norms = np.sum(x**2, axis=1)
K = np.exp(-0.5 * (norms[:, None] + norms[None, :] - 2 * x @ x.T))
K += 1e-3 * 1e-3 * np.eye(n)  # noise^2

# Write to temp files
Path("/tmp/test_native").mkdir(exist_ok=True)

# Write pre-built MATRIX (not features!)
K.tofile("/tmp/test_native/matrix_prebuilt.bin")

# Write targets
y.tofile("/tmp/test_native/targets.bin")

# Write metadata with input_kind="matrix" to force old path
meta = {
    "n": n,
    "d": 0,  # Not used for matrix input
    "input_kind": "matrix",  # Forces old centralized path!
    "length_scale": 1.0,
    "variance": 1.0,
    "noise": 1e-3
}
with open("/tmp/test_native/input_meta_old.json", "w") as f:
    json.dump(meta, f)

print(f"Pre-built matrix written to /tmp/test_native/")
print(f"n={n}, matrix shape: {K.shape}")
print(f"K[0,0]={K[0,0]:.6f}, K[0,1]={K[0,1]:.6f}")
EOF

# Run native code with OLD path (centralized matrix assembly)
echo ""
echo "========================================="
echo "Testing OLD ScaLAPACK Path (Centralized)"
echo "========================================="
echo ""
mpirun -np 1 native/build/scalapack_gp_fit \
  --input-meta /tmp/test_native/input_meta_old.json \
  --matrix-bin /tmp/test_native/matrix_prebuilt.bin \
  --rhs-bin /tmp/test_native/targets.bin \
  --output-meta /tmp/test_native/output_old.json \
  --alpha-bin /tmp/test_native/alpha_old.bin \
  --chol-bin /tmp/test_native/chol_old.bin \
  --backend scalapack \
  --block-size 64

echo ""
echo "Exit code: $?"
echo ""
if [ -f /tmp/test_native/output_old.json ]; then
  echo "SUCCESS! Old path works."
  echo "Output metadata:"
  cat /tmp/test_native/output_old.json
else
  echo "FAILED! Old path is also broken."
fi
