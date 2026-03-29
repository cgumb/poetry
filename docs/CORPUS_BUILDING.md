# Corpus building

The canonical corpus build path is:

```bash
python scripts/build_poetry_corpus.py
```

The active source foundation lives in:

```text
configs/poetry_sources.json
```

If you want to change which sources are used, edit that manifest rather than modifying scripts.

## Manifest format

Each source entry has:

- `name`: stable source name used in logs and `--sources`
- `kind`: one of `huggingface`, `parquet`, `csv`, `json`, `jsonl`, `ndjson`
- `location`: dataset ID for Hugging Face, or a local file path
- `split`: split name for Hugging Face datasets
- `enabled`: whether this source is active
- `priority`: lower values win during dedupe when duplicate poems appear in multiple sources
- `language`
- `license_family`

A richer template lives in:

```text
configs/poetry_sources.template.json
```

## Main commands

Build all enabled sources:

```bash
python scripts/build_poetry_corpus.py
```

Build only a subset:

```bash
python scripts/build_poetry_corpus.py --sources public_domain_poetry
python scripts/build_poetry_corpus.py --sources public_domain_poetry,gutenberg_extracted_poems
```

Cap ingest size per source for quick testing:

```bash
python scripts/build_poetry_corpus.py --per-source-limit 500
```

Adjust the minimum poem length retained:

```bash
python scripts/build_poetry_corpus.py --min-chars 80
```

## Normalization and dedupe

The corpus builder now keeps provenance columns and computes normalized keys before deduping.

Normalization currently includes:

- Unicode normalization (NFKC)
- standardizing curly quotes and dash variants
- trimming per-line whitespace
- collapsing repeated blank lines
- a looser whitespace-insensitive text normalization for duplicate detection

The default duplicate rule is:

1. keep one representative row per `text_hash_loose`
2. prefer lower `source_priority`
3. then prefer rows with richer metadata, especially non-empty `title` and `poet`
4. then prefer longer retained text
5. then use a stable tie-break by `poem_id`

This is intentionally simple and auditable. It is stronger than exact `(title, poet, text)` matching, but still conservative and inspectable.

## Outputs

The canonical build writes:

- `data/poems.parquet`: deduped canonical corpus
- `data/source_audit.parquet`: per-source ingest and canonicalization counts
- `data/duplicate_poems.parquet`: dropped duplicate rows only
- `data/duplicate_groups.parquet`: both kept and dropped rows for each duplicate group

## Why two duplicate outputs exist

`duplicate_poems.parquet` is compact and useful when you only care about what was removed.

`duplicate_groups.parquet` is the better audit artifact because it keeps the winning row next to the losing rows in the same duplicate group.

Useful columns in the detailed duplicate-group report include:

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

## Suggested workflow when adding a new source

1. add the source to `configs/poetry_sources.json`
2. run `python scripts/build_poetry_corpus.py --per-source-limit ...`
3. inspect `data/source_audit.parquet`
4. inspect `data/duplicate_groups.parquet`
5. only then enable the source for the full build

This keeps source expansion modular and makes duplicate decisions easier to verify.
