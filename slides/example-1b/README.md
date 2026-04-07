# Example 1b: Interactive CLI & Your Posterior

**Goal**: Rate poems, explore/exploit, and visualize your preference function.

## Prerequisites

- Completed Example 1a (environment set up)

## Overview

Use the interactive CLI to learn your poetry preferences through active learning.

## Step 1: Get Interactive Node (if needed)

If you exited from Example 1a:

```bash
cd poetry

# CPU node (sufficient for interactive use)
srun --pty -p general -N1 -n8 -t 60 bash

# OR GPU node (faster scoring with variance)
srun --pty -p gpu -N1 --gres=gpu:1 -t 60 bash
```

## Step 2: Activate Environment

```bash
cd poetry
source scripts/activate_env.sh              # CPU
source scripts/activate_env.sh --gpu        # GPU (if on GPU node)
```

## Step 3: Launch CLI

```bash
python scripts/app/interactive_cli.py
```

## Step 4: Build Your Preference Model

**Commands**:
- `n <rating>` - Rate current poem (-1 to 1, or 'skip')
- `e` - **Exploit**: Get recommendation
- `x` - **Explore**: Get informative query
- `v` - Visualize posterior
- `s` - Show session stats
- `q` - Quit

**Strategy**:

1. **Initial ratings** (5-10 poems): Rate honestly with your actual preferences
2. **Exploration** (`x`): Get informative poems (reduces uncertainty)
3. **Exploitation** (`e`): Get recommendations (tests learned preferences)
4. **Visualize** (`v`): Create posterior heatmap (after 20+ ratings)

## Step 5: Visualize Your Preferences

After rating 20+ poems:

```
v
```

This creates:
```
results/session_plots/latest_posterior_mean_hexbin.png
```

The visualization shows:
- **Hexbins**: Poem embedding space (2D UMAP projection)
- **Colors**: Your predicted preference (red=love, blue=dislike)
- **Black dots**: Poems you rated
- **Triangles**: Poet clusters

**Copy to view locally**:
```bash
# From your local machine:
scp <user>@<cluster>:poetry/results/session_plots/latest_*.png .
```

## Understanding the Heatmap

- **Hot regions** (red): Poems similar to ones you liked
- **Cold regions** (blue): Poems similar to ones you disliked
- **Smooth gradients**: GP interpolation between observations
- **Poet clusters**: Similar poets group in embedding space

## Tips

1. **Rate honestly** - model learns your actual preferences
2. **Explore early** (`x`) - better coverage of poem space
3. **Exploit later** (`e`) - test if recommendations match your taste
4. **Rate 20-30 poems** - more data → better posterior

## Example Session

```bash
# Rate a few poems
Rating: 0.8
Rating: -0.5
Rating: 0.3

# Explore to find informative poems
> x
[High-uncertainty poem shown]
Rating: 0.6

# Get recommendation
> e
[Similar to your likes]
Rating: 0.9

# Visualize (after ~20 ratings)
> v
✓ Saved posterior visualization

# Check stats
> s
Rated poems: 23
```

## Next Steps

If time permits, continue to **Example 1c** for benchmarking experiments.

Otherwise: **You're done!** You've experienced active learning with GPs on real data.
