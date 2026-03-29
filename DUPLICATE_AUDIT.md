# Duplicate audit workflow

The multi-source ingest path writes two different duplicate-oriented outputs:

- `data/duplicate_poems.parquet`: only the rows that were dropped
- `data/duplicate_groups.parquet`: both the kept representative row and the dropped rows for each duplicate group

## Why both exist

`duplicate_poems.parquet` is compact and useful when you only care about what was removed.

`duplicate_groups.parquet` is better for auditing because it keeps the winning row next to the losing rows in the same duplicate group.

## Recommended audit columns

The detailed duplicate-group report includes these fields when available:

- `text_hash_loose`
- `kept_representative`
- `dedupe_rank`
- `duplicate_group_size`
- `source_priority`
- `source_name`
- `poem_id`
- `source_row_id`
- `title`
- `poet`
- `text_len`
- `license_family`

## Basic usage

```bash
python scripts/build_poetry_corpus.py
```

To inspect only one source or a subset of sources:

```bash
python scripts/build_poetry_corpus.py --sources public_domain_poetry
python scripts/build_poetry_corpus.py --sources public_domain_poetry,gutenberg_extracted_poems
```

## Interpreting winners and losers

The current dedupe logic keeps one row per normalized loose-text hash and prefers:

1. lower `source_priority`
2. richer metadata (`title` / `poet` present)
3. longer retained text
4. stable tie-break by `poem_id`

So if two poems normalize to the same loose text, the group report should make it clear which source won and why.
