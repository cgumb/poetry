# Example 1a: Quickstart + 2D Toy GP

**Goal**: Get environment running, visualize simple 2D GP.

---

## Quickstart Commands

**On login node**:
```bash
# Clone repository
git clone https://github.com/cgumb/poetry.git
cd poetry

# Get interactive compute node (90 min)
srun --pty -p general -N1 -n8 -t 90 bash
```

**On compute node**:
```bash
cd poetry

# Bootstrap environment
bash scripts/bootstrap_venv.sh
source scripts/activate_env.sh

# Build native code
make native-build

# Get shared data (~100MB)
bash scripts/setup_shared_data.sh

# Verify setup
python -c "import poetry_gp; print('✓ poetry_gp installed')"
ls -lh data/shared/
ls -lh native/build/release/scalapack_gp_fit
```

**Expected**: `✓ poetry_gp installed`, data files, executable present.

---

## Visualize 2D GP

```bash
# Run toy GP example
python slides/example-1a/visualize_2d_gp.py

# Check output
ls -lh slides/example-1a/gp_2d_example.png
```

Creates visualization showing GP posterior mean with training points.

---

## Next

✅ Environment ready! Continue to **Example 1b** (interactive CLI).

---

## GPU Setup (Alternative)

If using GPU node instead:

```bash
# On login node
srun --pty -p gpu -N1 --gres=gpu:1 -t 90 bash

# On GPU node
cd poetry
bash scripts/bootstrap_venv.sh --gpu
source scripts/activate_env.sh --gpu
# Skip make native-build on GPU
bash scripts/setup_shared_data.sh

# Verify GPU
python -c "import cupy as cp; print(f'✓ CuPy {cp.__version__}')"
```
