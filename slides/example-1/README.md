# Example 1: Quickstart & Environment Setup

**Goal**: Get the Poetry GP environment running and verify everything works.

## Prerequisites

- SSH access to the cluster
- Basic familiarity with bash commands

## Step 1: Clone the Repository

From the **login node**, clone the repository:

```bash
git clone https://github.com/cgumb/poetry.git
cd poetry
```

## Step 2: Get an Interactive Compute Node

You need a compute node to run code (don't run on login node!).

**For CPU work** (most examples):
```bash
srun --pty -p general -N1 -n8 -t 90 bash
```

**For GPU work** (Example 3 and optional for Example 4):
```bash
srun --pty -p gpu -N1 --gres=gpu:1 -t 90 bash
```

This gives you 90 minutes on a compute node.

## Step 3: Bootstrap the Environment

**On the compute node**, set up the Python environment:

```bash
# CPU environment
bash scripts/bootstrap_venv.sh

# OR for GPU environment (if you requested a GPU node)
bash scripts/bootstrap_venv.sh --gpu
```

This will take a few minutes. It creates a Python virtual environment with all dependencies.

## Step 4: Activate the Environment

```bash
# CPU
source scripts/activate_env.sh

# OR for GPU
source scripts/activate_env.sh --gpu
```

## Step 5: Build Native Code (CPU only)

If you're on a CPU node, build the native LAPACK/ScaLAPACK code:

```bash
make native-build
```

**Note**: Skip this on GPU nodes - the native code is CPU-only.

## Step 6: Get Shared Data

Download the pre-computed embeddings and poems:

```bash
bash scripts/setup_shared_data.sh
```

This downloads ~100MB of data (poem embeddings, etc.).

## Step 7: Environment Check

Verify everything is working:

```bash
# Check Python environment
python -c "import poetry_gp; print('✓ poetry_gp installed')"

# Check data is available
ls -lh data/shared/

# Check native build (CPU only)
ls -lh native/build/release/scalapack_gp_fit

# For GPU: check CuPy
python -c "import cupy as cp; print(f'✓ CuPy {cp.__version__} with CUDA {cp.cuda.runtime.runtimeGetVersion()}')"
```

**Expected output**:
- `✓ poetry_gp installed`
- Data files in `data/shared/`
- Native executable (CPU) or CuPy version (GPU)

## Troubleshooting

**Problem**: `poetry_gp` not found
- **Solution**: Make sure you ran `source scripts/activate_env.sh`

**Problem**: Native build fails
- **Solution**: Make sure you're on a CPU node (not GPU) and have run bootstrap first

**Problem**: `setup_shared_data.sh` fails
- **Solution**: Check internet connectivity from compute node

## Next Steps

✅ **Environment is ready!**

Continue to **Example 2** for a small fitting benchmark and GP visualization.
