# Data status

The experimental dataset has not been constructed or frozen yet.

- `raw/scenarios.jsonl` will contain one reviewed matched scenario per line.
- `processed/prefixes.parquet` and `processed/prefixes.csv` will be generated.
- `audits/` will contain automatic and human-review artifacts.
- `FROZEN_DATASET.sha256` must be created only after manual review.

The `templates/` directory is only an authoring aid. Its placeholder object is
not an experimental scenario and must never be appended unchanged. Do not
treat the safe example embedded in `src/smoke_test.py` as a dataset item.

Run `scripts/01_validate_dataset.py` on partial drafts. Refresh human-review
templates only before entering review decisions. Freezing is an explicit,
separate operation that requires 32 reviewed scenarios, eight per domain.

Do not place labels or evidence rationales inside model-visible messages. Do
not edit the dataset after freezing it without a documented versioned
correction and a full rerun of every downstream artifact.
