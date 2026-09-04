# Data status

The experimental dataset has not been constructed or frozen yet.

- `raw/scenarios.jsonl` will contain one reviewed matched scenario per line.
- `processed/prefixes.parquet` and `processed/prefixes.csv` will be generated.
- `audits/` will contain structural, lexical and blinded-label audits.
- `FROZEN_DATASET.sha256` must be created only after manual review.

Do not treat the safe example embedded in `src/smoke_test.py` as a dataset item.
