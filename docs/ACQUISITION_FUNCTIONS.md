# Acquisition Functions for Exploration

## Overview

Poetry GP supports multiple acquisition functions for exploration. Each has different computational complexity and exploration behavior.

## Available Strategies

### 1. `max_variance` (Recommended for Large Scale)

**What it does**: Selects the point with highest posterior variance.

**Complexity**: O(n) - instant for any n

**When to use**:
- Large candidate sets (n > 10k)
- Real-time recommendation
- When iteration speed matters

**Exploration behavior**:
- Information-theoretically optimal for minimizing entropy
- Picks most uncertain point
- Simple, effective, fast

**Performance for n=85k**:
```
Select time: ~0.001 seconds
```

**Example**:
```python
result = run_blocked_step(
    embeddings,
    rated_indices,
    ratings,
    exploration_strategy="max_variance",
)
```

### 2. `spatial_variance` (Exact Spatial Diversity)

**What it does**: Selects the point that maximizes total variance reduction across all candidates, considering spatial correlation.

**Complexity**: O(n²) - expensive for large n

**When to use**:
- Small-medium candidate sets (n < 10k)
- When spatial diversity is critical
- Batch active learning (select multiple diverse points)
- When you can afford 8+ seconds per iteration

**Exploration behavior**:
- Considers full spatial correlation structure
- Promotes diverse exploration
- Avoids redundant nearby queries
- Better coverage of the space

**Performance**:
```
n=1k:   0.02 seconds
n=5k:   0.34 seconds
n=10k:  1.37 seconds
n=25k:  8.6 seconds
n=85k:  ~80 seconds (estimated)
```

**The cost**: For n=85k, computes 7.2 billion pairwise kernel evaluations.

**Example**:
```python
result = run_blocked_step(
    embeddings,
    rated_indices,
    ratings,
    exploration_strategy="spatial_variance",
)
```

**Note**: GPU acceleration was tested but actually **slower** than CPU due to memory-bound nature and transfer overhead. CPU with multi-threaded BLAS is already well-optimized for this operation.

### 3. `expected_improvement` (Balanced Exploit/Explore)

**What it does**: Balances exploitation (high mean) and exploration (high variance) via Expected Improvement acquisition function.

**Complexity**: O(n) - instant for any n

**When to use**:
- Bayesian optimization scenarios
- When you want automatic exploitation/exploration balance
- Standard benchmark comparisons

**Exploration behavior**:
- Classic Bayesian optimization acquisition function
- Balances mean and variance naturally
- Less exploratory than max_variance
- More focused on optimizing the objective

**Performance for n=85k**:
```
Select time: ~0.002 seconds
```

**Example**:
```python
result = run_blocked_step(
    embeddings,
    rated_indices,
    ratings,
    exploration_strategy="expected_improvement",
)
```

## Performance Comparison

| Strategy              | Complexity | n=1k    | n=10k   | n=85k     |
|-----------------------|-----------|---------|---------|-----------|
| max_variance          | O(n)      | 0.001s  | 0.001s  | 0.001s    |
| expected_improvement  | O(n)      | 0.002s  | 0.002s  | 0.002s    |
| spatial_variance      | O(n²)     | 0.02s   | 1.37s   | ~80s      |

## When Spatial Diversity Matters

`spatial_variance` is valuable when:

1. **Batch active learning**: Selecting k diverse points simultaneously
   ```python
   # Select 10 diverse exploration points
   diverse_indices = []
   for _ in range(10):
       result = run_blocked_step(..., exploration_strategy="spatial_variance")
       diverse_indices.append(result.explore_index)
       excluded_mask[result.explore_index] = True
   ```

2. **Coverage requirements**: Need to explore different regions of the space

3. **Small datasets**: n < 10k where O(n²) is acceptable

## When to Use What

### Large-scale recommendation (n=85k poems):
```python
# Use max_variance - instant, effective
exploration_strategy="max_variance"
```

### Medium-scale with quality focus (n=5k):
```python
# Can afford spatial_variance - better diversity
exploration_strategy="spatial_variance"  # 0.34s per iteration
```

### Bayesian optimization:
```python
# Expected improvement is standard
exploration_strategy="expected_improvement"
```

### Interactive CLI with 85k candidates:
```python
# max_variance for responsive UX
exploration_strategy="max_variance"  # 0.001s vs 80s
```

## Honest Tradeoffs

### For n=85k candidates:

**Option A: Fast exploration** (max_variance)
- Pro: 0.001s per iteration (instant)
- Pro: Information-theoretically optimal
- Con: No explicit spatial diversity

**Option B: Slow exploration** (spatial_variance)
- Pro: Explicit spatial diversity
- Pro: Better coverage guarantees
- Con: ~80s per iteration (8000× slower)

**Reality**: For interactive recommendation with 85k candidates, `max_variance` is the practical choice. `spatial_variance` is simply too expensive for this scale without algorithmic changes (low-rank approximations, different kernel structures, etc.).

## Configuration

Set in `GPConfig`:

```python
from poetry_gp.config import GPConfig

# Fast exploration (recommended for large n)
config = GPConfig(exploration_strategy="max_variance")

# Spatial diversity (only for small-medium n)
config = GPConfig(exploration_strategy="spatial_variance")

# Balanced exploit/explore
config = GPConfig(exploration_strategy="expected_improvement")
```

Or pass directly:

```python
result = run_blocked_step(
    embeddings,
    rated_indices,
    ratings,
    exploration_strategy="max_variance",  # or "spatial_variance", "expected_improvement"
)
```

## Summary

| Need                          | Strategy              | Time (n=85k) |
|-------------------------------|-----------------------|--------------|
| **Fast, interactive**         | max_variance          | 0.001s       |
| **Balanced**                  | expected_improvement  | 0.002s       |
| **Spatial diversity**         | spatial_variance      | ~80s         |

**Recommendation for Poetry GP with 85k poems**: Use `max_variance` for interactive exploration. The O(n²) cost of `spatial_variance` makes it impractical at this scale.
