# Example 4: Interactive CLI & Posterior Visualization

**Goal**: Use the interactive CLI to rate poems, explore/exploit, and visualize your learned preference function.

## Prerequisites

- Completed Example 1 (environment setup)
- Shared data downloaded

## Overview

The interactive CLI lets you:
- Rate poems on a scale (your true preferences)
- **Exploit**: Get recommendations based on learned preferences
- **Explore**: Get informative queries to improve the model
- **Visualize**: See your posterior preference function

## Step 1: Get an Interactive Compute Node

The CLI is interactive, so you need a compute node (not a batch job).

**From the login node**:
```bash
cd poetry

# CPU node (sufficient for interactive use)
srun --pty -p general -N1 -n8 -t 60 bash

# OR GPU node (for faster scoring with variance)
srun --pty -p gpu -N1 --gres=gpu:1 -t 60 bash
```

This gives you 60 minutes for interactive exploration.

## Step 2: Activate Environment

**On the compute node**:
```bash
cd poetry

# Activate appropriate environment
source scripts/activate_env.sh              # CPU
source scripts/activate_env.sh --gpu        # GPU
```

## Step 3: Launch Interactive CLI

```bash
python scripts/app/interactive_cli.py
```

You'll see:
```
============================================================
Poetry GP - Interactive Preference Learning
============================================================
Commands:
  n <rating>  - Rate current poem (-1 to 1, or skip)
  e           - Exploit: Get recommendation
  x           - Explore: Get informative query
  v           - Visualize posterior
  s           - Show session stats
  q           - Quit

Current poem:
============================================================
[Poem text appears here]
============================================================
Rating (-1 to 1, or 'skip'):
```

## Step 4: Build Your Preference Model

**Strategy**:

1. **Initial ratings** (5-10 poems):
   - Rate a few poems to bootstrap the GP
   - Use your actual preferences! (-1 = dislike, +1 = love, 0 = neutral)

2. **Exploration** (`x` command):
   - Ask the model to show you informative poems
   - These maximize uncertainty reduction
   - Helps the model learn faster

3. **Exploitation** (`e` command):
   - Get recommendations based on learned preferences
   - See what the model thinks you'll like!

4. **Visualize** (`v` command):
   - Create a posterior heatmap
   - See your preference function in 2D embedding space

## Step 5: Create Posterior Visualization

After rating 20+ poems, create a visualization:

```
v
```

The CLI will generate:
```
results/session_plots/latest_posterior_mean_hexbin.png
results/session_plots/latest_posterior_variance_hexbin.png
```

**View the files** (copy to local machine or use `scp`):
```bash
ls -lh results/session_plots/
```

The visualization shows:
- **Hexbins**: Poem embedding space (2D UMAP projection)
- **Colors**: Your predicted preference (red = love, blue = dislike)
- **Black dots**: Poems you rated
- **Triangles**: Poet clusters

## Understanding the Visualization

The heatmap reveals:
- **Hot regions** (red): Poems similar to those you liked
- **Cold regions** (blue): Poems similar to those you disliked
- **Smooth gradients**: GP interpolation
- **Poet clusters**: Similar poets group together in embedding space

## Tips for Good Results

1. **Rate honestly**: The model learns your actual preferences
2. **Explore early**: Use `x` to find diverse poems (better coverage)
3. **Exploit later**: Use `e` to test if recommendations match your taste
4. **Rate 20-30 poems**: More data → better posterior
5. **Visualize periodically**: See how your model evolves

## Session Statistics

Use `s` command to see:
- Number of rated poems
- Acquisition function used
- Model hyperparameters
- Backend performance (fit/score times)

## Example Session

```
# Rate a few poems with your honest opinion
Rating: 0.8
Rating: -0.5
Rating: 0.3

# Explore to find informative poems
> x
[System shows high-uncertainty poem]
Rating: 0.6

# Check recommendations
> e
[System recommends poem similar to your likes]
Rating: 0.9

# Visualize after ~20 ratings
> v
✓ Saved posterior visualization

# Check stats
> s
Rated poems: 23
Fit backend: native_lapack
Latest fit time: 0.03s
```

## Files Created

- `results/session_plots/latest_posterior_mean_hexbin.png`
- `results/session_plots/latest_posterior_variance_hexbin.png`
- Session state saved in `data/sessions/` (if configured)

## Troubleshooting

**Problem**: CLI is slow
- **Solution**: Use GPU node for faster scoring, or reduce candidate set

**Problem**: Visualization fails
- **Solution**: Make sure you've rated at least 10 poems first

**Problem**: "No module named poetry_gp"
- **Solution**: Run `source scripts/activate_env.sh` again

## Next Steps

✅ **You've completed all examples!**

**Further exploration**:
- Try different acquisition functions (edit `interactive_cli.py`)
- Run larger benchmarks with more processes
- Experiment with different kernel hyperparameters
- Compare your preference function with classmates

**Questions?** Check the main repository README or open an issue on GitHub.
