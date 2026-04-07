# Example 1a: Quickstart & Toy GP Visualization

**Goal**: Get environment running and see a simple 2D GP example.

## Step 1: Clone Repository

From the **login node**:

```bash
git clone https://github.com/cgumb/poetry.git
cd poetry
```

## Step 2: Get Interactive Compute Node

```bash
# CPU node (for this example)
srun --pty -p general -N1 -n8 -t 90 bash
```

## Step 3: Bootstrap Environment

**On the compute node**:

```bash
cd poetry

# Setup Python environment
bash scripts/bootstrap_venv.sh
source scripts/activate_env.sh

# Build native code
make native-build

# Get shared data
bash scripts/setup_shared_data.sh
```

This takes ~5 minutes.

## Step 4: Environment Check

```bash
# Check Python
python -c "import poetry_gp; print('✓ poetry_gp installed')"

# Check data
ls -lh data/shared/

# Check native build
ls -lh native/build/release/scalapack_gp_fit
```

Expected: `✓ poetry_gp installed`, data files, and executable present.

## Step 5: Visualize 2D GP

Run a toy 2D example to see how GPs work:

```bash
python slides/example-1a/visualize_2d_gp.py
```

This creates `slides/example-1a/gp_2d_example.png` showing:
- **Colors**: GP posterior mean (predicted function value)
- **Large dots**: Training observations
- **Contours**: Iso-lines of posterior mean

**Key insight**: The GP interpolates smoothly between observations, with uncertainty growing in unexplored regions.

## Troubleshooting

**Problem**: `poetry_gp` not found
- **Solution**: Run `source scripts/activate_env.sh`

**Problem**: Native build fails
- **Solution**: Make sure you're on CPU node (not GPU)

## Next Steps

✅ **Environment ready!**

Continue to **Example 1b** to play with the interactive CLI (rate poems, get recommendations, visualize your preferences).
