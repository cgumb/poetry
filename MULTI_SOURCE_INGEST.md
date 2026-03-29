# Multi-source poetry ingest

This adds a source-manifest-driven ingest path so the corpus foundation can be changed by editing a single JSON list.

## Manifest

Edit `configs/poetry_sources.json`.

Each entry has:

- `name`: stable source name used in logs and `--sources`
- `kind`: one of `huggingface`, `parquet`, `csv`, `json`, `jsonl`, `ndjson`
- `location`: dataset ID for Hugging Face, or a local file path
- `split`: split name for Hugging Face datasets
- `enabled`: whether this source is active
- `priority`: lower values win during dedupe if duplicate poems appear in multiple sources
- `language`
- `license_family`

## Main command

```bash
python scripts/fetch_prepare_multi_source_poetry.py
```

Useful options:

```bash
python scripts/fetch_prepare_multi_source_poetry.py --sources public_domain_poetry
python scripts/fetch_prepare_multi_source_poetry.py --manifest configs/poetry_sources.json --min-chars 80
python scripts/fetch_prepare_multi_source_poetry.py --per-source-limit 500
```

## Dedupe behavior

The new ingest path now keeps provenance columns and computes normalized/hashing keys before deduping.

Normalization includes:

- Unicode normalization (NFKC)
- standardizing curly quotes and dash variants
- trimming per-line whitespace
- collapsing repeated blank lines
- a looser whitespace-insensitive text normalization for duplicate detection

The default duplicate rule is:

- keep one representative row per `text_hash_loose`
- prefer lower `source_priority`
- then prefer rows with richer title/poet metadata
- then prefer longer retained text

Dropped duplicates are written to `data/duplicate_poems.parquet` for inspection.

## Output

The deduped dataset keeps the original core columns:

- `poem_id`
- `title`
- `poet`
- `text`

and adds provenance / audit columns such as:

- `source_name`
- `source_kind`
- `source_location`
- `source_split`
- `license_family`
- `source_row_id`
- `text_hash`
- `text_hash_loose`
- `title_poet_text_key`
