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

### Strategy 2: Embedding Similarity

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
- Only works for poets with sufficient known poems
- Only works for poems stylistically similar to known work
- High threshold needed to avoid false positives

### Strategy 3: LLM Batch API (Optional)

**Approach**: For remaining uncertain cases, use Claude's batch API to identify poets from text.

**Algorithm**:
1. Generate JSONL batch file with prompts for each poem
2. Submit to Claude batch API (50% cost reduction vs real-time)
3. Parse JSON responses, apply only high-confidence identifications

**Example prompt**:
```
Given this poem excerpt, identify the poet and title if recognizable.

Poem excerpt:
Because I could not stop for Death –
He kindly stopped for me –
The Carriage held but just Ourselves –
And Immortality.

[Respond with JSON: {"poet": "...", "title": "...", "confidence": "high|medium|low"}]
```

**Cost**: ~$0.25 per 1000 poems (using Haiku, batch pricing)

**Advantages**:
- Can identify poems LLM was trained on
- Provides title imputation as well
- Higher accuracy than similarity for well-known poems

**Limitations**:
- Costs money (though batch API is cheap)
- Only works for poems in LLM training data
- Requires API access and async processing

### Usage

**Basic imputation** (first-line + embedding similarity):
```bash
python scripts/app/impute_missing_metadata.py \
  --poems data/poems.parquet \
  --embeddings data/embeddings.npy \
  --output data/poems_imputed.parquet \
  --similarity-threshold 0.95 \
  --min-known-poems 5
```

**Dry run** (see what would be imputed):
```bash
python scripts/app/impute_missing_metadata.py --dry-run
```

**Generate LLM batch file** (for remaining uncertain cases):
```bash
# After running imputation, edit script to add --generate-llm-batch flag
# Then submit generated JSONL to Claude batch API
# See: https://docs.anthropic.com/en/docs/build-with-claude/message-batching
```

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
