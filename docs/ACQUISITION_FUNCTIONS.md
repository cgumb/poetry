# Acquisition Functions for GP-Based Recommendation

This document describes the acquisition functions implemented for exploit/explore recommendations in the poetry GP system.

## Overview

An **acquisition function** determines which point to select next based on the GP posterior. Different functions optimize different objectives.

## Exploitation Strategies (Finding Best Items)

Used when running the **exploit** command - recommending poems you're likely to enjoy.

### 1. Max Mean (Simple)

**Formula:** `argmax μ(x)`

**What it does:**
- Picks the poem with the highest predicted rating
- Ignores uncertainty completely

**When to use:**
- When you trust the model (many ratings, low uncertainty)
- Speed is critical
- You want the "safest bet" based on current knowledge

**Limitations:**
- **Risky for uncertain poems**: A poem might have μ=4.5 but σ=2.0 (could be anywhere from 2.5 to 6.5!)
- Ignores exploration entirely

**Complexity:** O(n) - just find max of mean array

---

### 2. UCB - Upper Confidence Bound (RECOMMENDED)

**Formula:** `argmax μ(x) + β·σ(x)`

**What it does:**
- Balances predicted quality (μ) and uncertainty (σ)
- Higher β = more optimistic/exploratory
- Industry standard for recommendation systems

**Parameters:**
- `β = 1.0`: More exploitation (trust the mean)
- `β = 2.0`: **Balanced (recommended default)**
- `β = 3.0`: More exploration (favor uncertainty)

**When to use:**
- **Default choice for most applications**
- When you want confident recommendations
- When avoiding bad recommendations is important

**Example:**
```
Poem A: μ=4.0, σ=0.5  →  UCB = 4.0 + 2.0×0.5 = 5.0
Poem B: μ=3.5, σ=1.0  →  UCB = 3.5 + 2.0×1.0 = 5.5 ← Selected!
```

Poem B is selected because its uncertainty gives it potential upside despite lower mean.

**Theory:** UCB has **theoretical guarantees** for minimizing regret in multi-armed bandit problems.

**Complexity:** O(n) - compute mean + β·std for all points

---

### 3. LCB - Lower Confidence Bound (Conservative)

**Formula:** `argmax μ(x) - β·σ(x)`

**What it does:**
- Pessimistic/conservative recommendation
- Picks points with high "worst-case" quality
- Avoids risky uncertain poems

**When to use:**
- When avoiding bad recommendations is critical
- Conservative exploration
- High-stakes scenarios

**Example:**
```
Poem A: μ=4.0, σ=0.5  →  LCB = 4.0 - 2.0×0.5 = 3.0
Poem B: μ=3.5, σ=1.0  →  LCB = 3.5 - 2.0×1.0 = 1.5
```

Poem A is selected because it has a better "worst case" (3.0 vs 1.5).

**Complexity:** O(n)

---

### 4. Thompson Sampling (Bayesian Optimal)

**Formula:** Sample `f(x) ~ N(μ(x), σ²(x))` for all x, return `argmax f(x)`

**What it does:**
- Randomly samples from the posterior distribution
- Returns the max of the sampled function
- Naturally explores uncertain regions

**When to use:**
- Want diverse recommendations across sessions
- Theoretical optimality for bandits
- More "natural" exploration than deterministic methods

**Properties:**
- **Bayesian optimal** for independent arms
- **Stochastic** - different recommendations each time
- **Exploration proportional to uncertainty** - naturally balances exploit/explore

**Example:**
```
Poem A: μ=4.0, σ=0.5  →  Sample: 3.8
Poem B: μ=3.5, σ=1.0  →  Sample: 4.2 ← Selected this time!
```

Next time, Poem A might be selected if its sample is higher.

**Complexity:** O(n) - sample from n Gaussians

---

## Exploration Strategies (Learning Preferences)

Used when running the **explore** command - discovering poems to learn your preferences.

### 1. Max Variance (Information-Theoretic Optimal)

**Formula:** `argmax σ²(x)`

**What it does:**
- Picks the poem with highest posterior variance
- **Minimizes posterior entropy** H(f|D)
- Information-theoretically optimal for GP exploration

**Theory:**
For a GP, the information gain from observing y at x is:
```
IG(x) = H(f|D) - E[H(f|D ∪ {x,y})] = (1/2) log(σ²(x) + σₙ²)
```

This is monotonic in σ²(x), so **max variance = max information gain**!

**When to use:**
- **Default exploration strategy** - provably optimal
- Fast - O(1) since variance already computed
- Large candidate sets (n > 10k)

**Limitations:**
- Ignores spatial correlation between points
- May repeatedly query isolated high-variance regions

**Complexity:** O(n) - already computed, just find max

---

### 2. Spatial Diverse (Mean Variance Reduction)

**Formula:** `argmax Σᵢ k(xᵢ, x*)² / (σ²(x*) + σₙ²)`

**What it does:**
- Picks the point that would **reduce total uncertainty** across ALL candidates
- Considers spatial correlation via kernel k(·,·)
- More spatially diverse exploration than max variance

**Theory:**
Observing y* at x* updates variance at another point xᵢ as:
```
σ²_new(xᵢ) = σ²(xᵢ) - k(xᵢ, x*)² / (σ²(x*) + σₙ²)
```

Summing across all points gives total variance reduction.

**When to use:**
- Exploration quality > speed
- Want spatially diverse coverage
- Smaller candidate sets (n < 5k)

**Limitations:**
- **Expensive**: O(n² × d) for pairwise kernel + O(n²) for scoring
- For n=10k, that's 100M kernel evaluations!

**Example:**
```
Point A: σ²=2.0, but isolated (low k(xᵢ, A) for most i)
  → SVR(A) = small (doesn't tell us much about other points)

Point B: σ²=1.5, but central (high k(xᵢ, B) for many i)
  → SVR(B) = large (tells us about many nearby points)
```

Point B is selected for better spatial coverage.

**Complexity:** O(n² × d) - requires pairwise kernel matrix

---

### 3. Expected Improvement (Balanced)

**Formula:** `EI(x) = (μ(x) - f_best) · Φ(Z) + σ(x) · φ(Z)`

where:
- `Z = (μ(x) - f_best) / σ(x)`
- `Φ = standard normal CDF`
- `φ = standard normal PDF`
- `f_best = max observed rating so far`

**What it does:**
- Balances exploitation (find better than current best) and exploration (high uncertainty)
- Classic acquisition function from **Bayesian optimization**
- Measures expected amount by which x improves over current best

**Theory:**
EI integrates over the posterior to compute expected improvement:
```
EI(x) = ∫_{-∞}^{∞} max(0, y - f_best) · N(y | μ(x), σ²(x)) dy
```

For Gaussian posterior, this has a closed-form solution.

**When to use:**
- Want balanced explore/exploit in one command
- Classic choice in Bayesian optimization
- When you care about improving over current best

**Properties:**
- **Acquisition**: Returns 0 if μ(x) ≪ f_best and σ(x) ≈ 0 (no hope)
- **Exploration**: Increases with σ(x)
- **Exploitation**: Increases with μ(x)

**Example:**
```
Current best rating: f_best = 4.0

Poem A: μ=4.5, σ=0.5
  Z = (4.5-4.0)/0.5 = 1.0
  EI = 0.5·Φ(1.0) + 0.5·φ(1.0) = 0.5·0.841 + 0.5·0.242 = 0.542

Poem B: μ=3.0, σ=2.0
  Z = (3.0-4.0)/2.0 = -0.5
  EI = -1.0·Φ(-0.5) + 2.0·φ(-0.5) = -1.0·0.309 + 2.0·0.352 = 0.395
```

Poem A selected (higher EI despite lower variance).

**Complexity:** O(n) - evaluate EI formula for each point

---

## Computational Complexity Summary

| Strategy | Complexity | Notes |
|----------|------------|-------|
| **Max Mean** | O(n) | Requires mean only (can skip variance!) |
| **UCB/LCB** | O(n) | Requires both mean and variance |
| **Thompson** | O(n) | Requires both mean and variance |
| **Max Variance** | O(1) | Already computed, just find max |
| **Spatial Diverse** | O(n² × d) | **Very expensive** - pairwise kernel |
| **Expected Improvement** | O(n) | Requires both mean and variance |

## Optional Variance Computation

Variance computation is **expensive**: O(n × m²) due to triangular solve, while mean is only O(n × m).

**Optimization:** Use `compute_variance=False` when:
- Running **exploit with Max Mean** (doesn't need variance)
- m is large (variance becomes O(m²) bottleneck)

**Speedup:**
- m=100: ~1.5× faster
- m=10,000: ~100× faster!

**Implementation:**
```python
result = run_blocked_step(
    embeddings, rated_indices, ratings,
    exploitation_strategy="max_mean",
    compute_variance=False,  # Skip expensive variance computation
)
```

## Strategy Selection Guide

| Goal | Exploitation | Exploration |
|------|--------------|-------------|
| **Safe recommendations** | UCB (β=2.0) | Max Variance |
| **Fastest** | Max Mean | Max Variance |
| **Most diverse** | Thompson | Spatial Diverse |
| **Best theory** | UCB | Max Variance |
| **Balanced explore+exploit** | UCB (β=2.0) | Expected Improvement |
| **Conservative** | LCB | Max Variance |
| **Spatially aware** | — | Spatial Diverse |

## References

- **UCB/Thompson**: Slivkins, "Introduction to Multi-Armed Bandits" (2019)
- **Expected Improvement**: Jones et al., "Efficient Global Optimization of Expensive Black-Box Functions" (1998)
- **GP Acquisition Functions**: Shahriari et al., "Taking the Human Out of the Loop" (2016)
- **Information Gain**: Krause & Guestrin, "Near-Optimal Sensor Placements in Gaussian Processes" (2005)
