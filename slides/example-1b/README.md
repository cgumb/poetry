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

**Basic usage (no visualization)**:
```bash
python scripts/app/interactive_cli.py
```

**With visualization support** (requires 2D projection):
```bash
python scripts/app/interactive_cli.py \
  --coords-2d data/coords_2d.npy \
  --viz-output-dir data/viz
```

## Step 4: Build Your Preference Model

**Commands**:
- `l` - **Like** current poem (+1.0)
- `n` - **Neutral** (0.0)
- `d` - **Dislike** (-1.0)
- `e` - **Exploit**: Get recommendation
- `x` - **Explore**: Get informative query
- `v` - **Visualize** posterior heatmaps (requires --coords-2d)
- `s` - **Search** poems by title/poet/text
- `r` - Show **rated** poems
- `c` - Open **config** menu
- `q` - **Quit**

**Strategy**:

1. **Initial ratings** (5-10 poems): Rate honestly with your actual preferences
2. **Exploration** (`x`): Get informative poems (reduces uncertainty)
3. **Exploitation** (`e`): Get recommendations (tests learned preferences)
4. **Visualize** (`v`): Create posterior heatmap (after 20+ ratings)

## Step 5: Visualize Your Preferences

After rating 5+ poems, use the `v` command to generate posterior heatmaps:

```
v
```

This creates two visualizations in `data/viz/`:
```
data/viz/latest_posterior_mean.png       # Predicted ratings
data/viz/latest_posterior_variance.png   # Uncertainty map
```

The CLI will display **clickable file:// links** (in supported terminals) for easy viewing!

**Visualization shows**:
- **Heatmap**: Smoothed posterior over 2D poem space (UMAP projection)
- **Colors**:
  - Mean: Red/blue diverging (red=predicted like, blue=predicted dislike)
  - Variance: Yellow/orange sequential (bright=uncertain, dark=confident)
- **Markers**:
  - Black circles: Poems you rated
  - Cyan X: Current poem
  - Green star: Exploit recommendation
  - Gold square: Explore recommendation
- **Purple triangles**: Poet centroids (if provided)

**Copy to view locally**:
```bash
# From your local machine:
scp <user>@<cluster>:poetry/data/viz/latest_*.png .
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
> l     # Like current poem (+1.0)
> d     # Dislike next poem (-1.0)
> n     # Neutral (0.0)

# Explore to find informative poems
> x
[High-uncertainty poem shown]
> l

# Get recommendation
> e
[Poem similar to your likes]
> l

# Visualize (after ~5+ ratings)
> v
✓ Visualization complete!
  📊 Posterior mean:     file:///.../data/viz/latest_posterior_mean.png
  📊 Posterior variance: file:///.../data/viz/latest_posterior_variance.png

# View rated poems
> r
[Table of all rated poems with scores]

# Search for specific poet
> s
🔍 Search title/poet/text: shakespeare
[Search results shown]
```

## Next Steps

If time permits, continue to **Example 1c** for benchmarking experiments.

Otherwise: **You're done!** You've experienced active learning with GPs on real data.
