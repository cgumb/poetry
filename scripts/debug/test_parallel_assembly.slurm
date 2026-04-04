#!/bin/bash
#SBATCH -p general
#SBATCH -N1
#SBATCH -n4
#SBATCH -t 00:05:00
#SBATCH -J test-parallel
#SBATCH -o results/test-parallel-%j.out
#SBATCH -e results/test-parallel-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/poetry}"
SPACK_SETUP_SCRIPT="${SPACK_SETUP_SCRIPT:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050}"

cd "$REPO_DIR"

echo "========================================"
echo "Test Parallel Assembly (4 processes)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Spack environment
if [[ -f "$SPACK_SETUP_SCRIPT" ]]; then
  source "$SPACK_SETUP_SCRIPT"
  spack env activate "$SPACK_ENV_NAME"
fi

# Disable CPU binding to avoid conflicts
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_mapping_policy=slot

# Create test data
python3 << 'EOF'
import numpy as np
import json
from pathlib import Path

np.random.seed(42)
n = 1000
d = 384

x = np.random.randn(n, d).astype(np.float64)
x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
y = np.random.randn(n).astype(np.float64)

Path("/tmp/test_parallel").mkdir(exist_ok=True)
x.tofile("/tmp/test_parallel/features.bin")
y.tofile("/tmp/test_parallel/targets.bin")

meta = {"n": n, "d": d, "input_kind": "features", "length_scale": 1.0, "variance": 1.0, "noise": 1e-3}
with open("/tmp/test_parallel/input_meta.json", "w") as f:
    json.dump(meta, f)

print(f"Created test data: n={n}, d={d}")
EOF

echo ""
echo "Running with 4 processes to verify parallel assembly:"
echo ""

# Use mpirun flags to explicitly disable binding
mpirun -np 4 --bind-to none --map-by slot native/build/scalapack_gp_fit \
  --input-meta /tmp/test_parallel/input_meta.json \
  --matrix-bin /tmp/test_parallel/features.bin \
  --rhs-bin /tmp/test_parallel/targets.bin \
  --output-meta /tmp/test_parallel/output.json \
  --alpha-bin /tmp/test_parallel/alpha.bin \
  --chol-bin /tmp/test_parallel/chol.bin \
  --backend scalapack \
  --block-size 64

echo ""
echo "Check above for '[DEBUG] Rank (X,Y) building local blocks' from ALL 4 ranks"
echo "If you only see rank (0,0), parallel assembly is NOT working!"
