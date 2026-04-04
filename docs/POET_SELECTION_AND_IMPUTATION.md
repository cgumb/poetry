# Poet Selection and Metadata Imputation

This document describes two data quality improvements:

1. **Canonical poet prioritization** in visualizations
2. **Missing metadata imputation** using multiple strategies

## Problem 1: Poet Visualization Selection

### The Issue

Visualizations previously selected poets purely by **poem count** in the corpus, causing:
- Prolific but obscure poets dominating the visualization
- Canonical poets (Dickinson, Yeats, Auden, Larkin, Heaney) being hidden
- Reduced pedagogical value

### Solution: Hybrid Selection Strategy

**Implementation**: `src/poetry_gp/canonical_poets.py` + updated visualization code

The new approach uses a **two-tier selection**:

1. **Tier 1**: Prioritize ~60 canonical poets from the curated list
   - Includes major poets from medieval to contemporary periods
   - Covers British, American, and international traditions
   - Poets marked as canonical get:
     - **Darker color** (darkviolet vs purple)
     - **Larger markers** (1.3× size boost)
     - **Higher opacity** (0.4 vs 0.2)
     - **Priority labeling** (labeled before high-count poets)

2. **Tier 2**: Fill remaining slots with high-count poets
   - Ensures corpus-specific patterns still visible
   - Helps identify dataset biases or interesting non-canonical clusters

### Visual Differences

**Old behavior**:
```python
# Sort by poem count only
order = np.argsort(-poets["n_poems"])
```

**New behavior**:
```python
# Split into canonical and non-canonical
canonical_order = canonical_indices[np.argsort(-n_poems[canonical_indices])]
non_canonical_order = non_canonical_indices[np.argsort(-n_poems[non_canonical_indices])]

# Prioritize canonical, then high-count
order = np.concatenate([canonical_order, non_canonical_order])
```

### Canonical Poet List

The list includes ~60 poets spanning:
- **Medieval & Renaissance**: Chaucer, Shakespeare, Donne, Milton
- **18th Century**: Pope, Blake
- **Romantic**: Wordsworth, Coleridge, Byron, Shelley, Keats
- **Victorian**: Tennyson, Browning, Hopkins, Hardy
- **American 19th**: Whitman, Dickinson, Poe
- **Modernist**: Yeats, Eliot, Pound, Frost, Stevens
- **Mid-20th British**: Auden, Dylan Thomas, Larkin, Hughes, Plath, Heaney
- **Mid-20th American**: Lowell, Bishop, Berryman, Brooks, Ginsberg
- **Contemporary**: Angelou, Rich, Walcott, Glück, Oliver

### Extending the List

To add poets to the canonical list:

```python
# Edit src/poetry_gp/canonical_poets.py
CANONICAL_POETS = {
    # ... existing poets ...
    "your new poet",  # lowercase, normalized
}
```

---

## Problem 2: Missing Poet Names and Titles

### The Issue

Poetry corpora often have:
- **Missing poet attribution** (anonymous, unknown, traditional)
- **Missing or generic titles** (untitled, poem)
- **Inconsistent metadata quality** across sources

This reduces:
- Search effectiveness
- User experience
- Corpus utility for recommendations

### Solution: Multi-Strategy Imputation

**Implementation**: `scripts/app/impute_missing_metadata.py`

### Strategy 1: First-Line Matching

**Approach**: Group poems by identical first lines, propagate known poets to unknowns.

**Rationale**: Same first line often means same poem (or very close variant).

**Example**:
```
"Shall I compare thee to a summer's day?"
→ If one instance has poet="William Shakespeare", propagate to others
```

**Cost**: Free (pure rule-based)

**Limitations**: Only works for poems with known duplicates in corpus

### Strategy 2: Embedding Similarity (OPTIONAL, NOT RECOMMENDED)

**Status**: Available but **disabled by default** due to robustness concerns.

**Approach**: Compute poet centroids from known poems, match unknown poems to nearest centroid.

**Algorithm**:
1. For each poet with ≥N known poems (default: N=5):
   - Compute mean embedding (centroid) from their known works
   - Normalize to unit vector
2. For each poem with missing poet:
   - Compute cosine similarity to all poet centroids
   - If max similarity ≥ threshold (default: 0.95), assign that poet
   - Store confidence score

**Example**:
```
Unknown poem with style similar to Emily Dickinson's known works
→ Cosine similarity = 0.97 to Dickinson centroid
→ Impute poet="Emily Dickinson", confidence=0.97
```

**Cost**: Free (uses existing embeddings)

**Limitations**:
- **Questionable robustness**: Poet style is complex and multidimensional
- Average embeddings may not capture poet-specific characteristics well
- Only works for poets with sufficient known poems
- High threshold needed to avoid false positives
- Can incorrectly attribute poems in similar styles

**To enable**: Add `--use-similarity` flag (use with caution)

### Strategy 3: LLM Batch API (Optional)

**Approach**: For remaining uncertain cases, use Claude's batch API to identify poets and titles from text.

**Complete Workflow**:

#### Step 1: Install Anthropic CLI (optional dependency)
```bash
source .venv/bin/activate
pip install -r requirements-llm.txt
# OR: pip install anthropic
```

#### Step 2: Generate batch request file
```bash
python scripts/app/impute_missing_metadata.py \
  --poems data/poems_imputed.parquet \
  --generate-llm-batch data/llm_batch_requests.jsonl \
  --max-llm-requests 1000  # Optional: limit for cost control
```

**Output**:
```
✓ Generated 247 batch API requests in data/llm_batch_requests.jsonl
  Estimated cost: ~$0.062
  (Based on ~43,225 input + ~12,350 output tokens)
```

#### Step 3: Submit to Claude Batch API
```bash
# Set your API key
export ANTHROPIC_API_KEY='your-api-key-here'

# Submit the batch
anthropic messages batches create --input-file data/llm_batch_requests.jsonl

# Returns: batch_abc123... (save this ID)
```

#### Step 4: Check batch status
```bash
# Check if complete (batches typically take minutes to hours)
anthropic messages batches retrieve batch_abc123...

# Status will show: in_progress → processing → ended
```

#### Step 5: Download results
```bash
anthropic messages batches results batch_abc123... > data/llm_batch_results.jsonl
```

#### Step 6: Apply results to dataframe
```bash
python scripts/app/apply_llm_imputation_results.py \
  --poems data/poems_imputed.parquet \
  --results data/llm_batch_results.jsonl \
  --output data/poems_imputed.parquet \
  --min-confidence medium
```

**Output**:
```
=== Results ===
Total LLM results processed: 247
Parse errors: 2
Skipped (already imputed): 0
Skipped (low confidence): 18

Applied imputations:
  Poet only: 89
  Title only: 34
  Both poet and title: 104
  Total poet imputations: 193
  Total title imputations: 138

✓ Writing updated poems to data/poems_imputed.parquet
```

**Example prompt sent to Claude**:
```
Given this poem excerpt, identify the poet and title if recognizable.

Poem excerpt:
Because I could not stop for Death –
He kindly stopped for me –
The Carriage held but just Ourselves –
And Immortality.

If you can confidently identify this poem, respond with JSON:
{"poet": "Emily Dickinson", "title": "Because I could not stop for Death", "confidence": "high"}

Respond only with valid JSON, no other text.
```

**Cost**: ~$0.003-0.005 per 1000 poems (using Haiku with batch pricing: 50% discount)
- Input tokens: $0.25 per 1M → $0.125 per 1M with batch
- Output tokens: $1.25 per 1M → $0.625 per 1M with batch

**Advantages**:
- Can identify poems LLM was trained on (classic/canonical works)
- Provides title imputation as well as poet
- Higher accuracy than similarity for well-known poems
- Batch API is 50% cheaper than real-time
- Async processing doesn't block other work

**Limitations**:
- Costs money (though very cheap with batch API)
- Only works for poems in LLM training data
- Requires Anthropic API access
- Async processing (minutes to hours delay)
- Won't work for very modern or obscure poems

**Safety features**:
- Only generates requests for rows still needing imputation
- Won't overwrite existing `poet_imputed=True` or `title_imputed=True` rows
- Confidence filtering (low/medium/high) to avoid bad imputations
- Dry-run mode to preview changes before applying

### Usage

**Complete recommended workflow**:

```bash
# Step 1: Install optional LLM dependencies (if using Claude Batch API)
pip install -r requirements-llm.txt

# Step 2: Run first-line matching imputation
python scripts/app/impute_missing_metadata.py \
  --poems data/poems.parquet \
  --output data/poems_imputed.parquet

# Step 3: Generate LLM batch file for remaining unknowns
python scripts/app/impute_missing_metadata.py \
  --poems data/poems_imputed.parquet \
  --generate-llm-batch data/llm_batch_requests.jsonl \
  --max-llm-requests 1000  # Optional: limit for cost control

# Step 4: Submit to Claude Batch API
export ANTHROPIC_API_KEY='your-api-key-here'
anthropic messages batches create --input-file data/llm_batch_requests.jsonl
# Returns: batch_abc123... (save this ID)

# Step 5: Check status (wait for completion)
anthropic messages batches retrieve batch_abc123...

# Step 6: Download results when complete
anthropic messages batches results batch_abc123... > data/llm_batch_results.jsonl

# Step 7: Apply results back to dataframe
python scripts/app/apply_llm_imputation_results.py \
  --poems data/poems_imputed.parquet \
  --results data/llm_batch_results.jsonl \
  --output data/poems_imputed.parquet \
  --min-confidence medium
```

**Running the apply script standalone**:
```bash
# Apply with different confidence thresholds
python scripts/app/apply_llm_imputation_results.py \
  --poems data/poems_imputed.parquet \
  --results data/llm_batch_results.jsonl \
  --output data/poems_imputed.parquet \
  --min-confidence high  # Only apply high-confidence results

# Dry run to preview changes
python scripts/app/apply_llm_imputation_results.py \
  --poems data/poems_imputed.parquet \
  --results data/llm_batch_results.jsonl \
  --output data/poems_imputed.parquet \
  --dry-run
```

**Dry run** (see what would be imputed without writing):
```bash
python scripts/app/impute_missing_metadata.py --poems data/poems.parquet --dry-run
```

**With embedding similarity** (not recommended, use at own risk):
```bash
python scripts/app/impute_missing_metadata.py \
  --poems data/poems.parquet \
  --embeddings data/embeddings.npy \
  --output data/poems_imputed.parquet \
  --use-similarity \
  --similarity-threshold 0.95
```

**Key safeguards**:
- ✅ **Never re-imputes** rows where `poet_imputed=True`
- ✅ **Only generates LLM requests** for rows still needing imputation
- ✅ **Cost control** via `--max-llm-requests` flag
- ✅ **Preserves existing data** when run multiple times

### Expected Results

Typical imputation rates (varies by corpus quality):
- **First-line matching**: 5-15% of missing poets
- **Embedding similarity**: 20-40% of missing poets (threshold=0.95)
- **Combined**: 25-50% of missing poets imputed with high confidence
- **LLM batch**: Can identify an additional 30-60% of remaining (if well-known poems)

### Quality Control

The script adds a `poet_imputation_confidence` column to track:
- **0.0**: Original data (not imputed)
- **> 0.0**: Imputed via embedding similarity (cosine similarity score)
- **Can be used for filtering**: Keep only high-confidence imputations

### Integration with Visualization

After imputation, rebuild poet centroids:
```bash
python scripts/app/build_poet_centroids.py \
  --poems data/poems_imputed.parquet \
  --embeddings data/embeddings.npy
```

The canonical poet prioritization will then show newly-identified canonical poets in visualizations.

### Data Schema

The imputation script adds these columns to the output:

- **`poet_imputed`** (bool): `True` if poet was imputed, `False` if original data
- **`poet_imputation_confidence`** (float): 0.0 for original data, or cosine similarity score (0-1) for imputed
- **`title_imputed`** (bool): Reserved for future title imputation (currently always `False`)

This allows downstream analysis to:
- Filter out low-confidence imputations
- Distinguish original vs inferred metadata
- Audit imputation quality

---

## Design Philosophy

Both solutions follow the principle: **Combine automated methods with curated knowledge**

1. **Canonical poet list**: Human-curated literary canon + automated poem-count ranking
2. **Metadata imputation**: Automated similarity matching + optional LLM verification

This hybrid approach:
- Leverages domain expertise where it matters (canonical poet selection)
- Uses automation for scale (similarity-based imputation)
- Provides opt-in expensive methods for high-value cases (LLM batch API)
- Maintains auditability (confidence scores, dry-run mode)

---

## Future Improvements

### Poet Selection
- **Spatial diversity**: Prefer poets spread across 2D projection space
- **Dynamic importance**: Weight by citation counts, anthology appearances
- **User preferences**: Let users specify custom canonical lists

### Metadata Imputation
- **Title imputation**: Extend to missing titles (similar strategies apply)
- **Multi-source consensus**: Cross-reference multiple corpora
- **Active learning**: Human verification of uncertain cases
- **Fuzzy matching**: Handle name variants (T.S. Eliot vs Thomas Stearns Eliot)

### Integration
- **Real-time imputation**: In corpus build pipeline
- **Confidence-based filtering**: Hide low-confidence imputations in UI
- **Provenance tracking**: Record imputation source and confidence
