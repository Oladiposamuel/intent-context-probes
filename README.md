# Intent or Topic Risk?

This repository implements the MATS application experiment described in
[`MATS_INTENT_PROBE_PROJECT_BRAIN.md`](MATS_INTENT_PROBE_PROJECT_BRAIN.md).
It tests whether a general Qwen3 chat model and its SafeRL counterpart linearly
encode observable evidence of harmful intent across matched multi-turn
conversations.

## Current milestone

Milestone 1 is complete:

- reproducible experiment configuration;
- Colab/GPU environment validation;
- Qwen model revision and metadata capture;
- shared chat-template rendering;
- assistant-decision-position activation extraction;
- a safe, built-in Qwen3-4B smoke test;
- local unit tests that do not download a model;
- a thin Colab driver notebook.

Milestone 2 dataset infrastructure is implemented:

- strict canonical scenario-schema validation;
- partial-draft validation without accidental freezing;
- structural, length and lexical audits;
- blinded-label and pairwise-semantic human-review sheets;
- explicit deterministic freezing with SHA-256 verification;
- immutable-change detection and documented revision handling;
- full-history/current-message prefix construction;
- pair, label and row-count invariants with local unit tests.

The real 32 scenarios have not been authored or frozen. Bulk model execution,
response annotation, baselines, probe training and evaluation remain later
milestones.

## Repository rules

- `configs/experiment.yaml` is authoritative for executable settings.
- `MATS_INTENT_PROBE_PROJECT_BRAIN.md` is authoritative for scientific design.
- Do not train or modify Qwen. Only small downstream classifiers are trained.
- Do not commit model weights, Hugging Face caches or activation arrays.
- Save costly-to-reproduce artifacts to persistent storage immediately.
- Use observable evidence of harmful intent, never claims about true intent.

## Google Colab quick start

Use a T4, L4 or A100 GPU with approximately 16 GB or more VRAM. In Colab,
mount Drive and set a persistent destination:

```python
from google.colab import drive
from pathlib import Path
import os

drive.mount("/content/drive")
persistent_root = Path("/content/drive/MyDrive/MATS_INTENT_PROBE/artifacts")
persistent_root.mkdir(parents=True, exist_ok=True)
os.environ["MATS_PERSISTENT_ARTIFACT_ROOT"] = str(persistent_root)
```

After loading a repository-scoped token into the `GH_TOKEN` environment
variable, clone the private repository and install dependencies:

```bash
gh auth setup-git
gh repo clone Oladiposamuel/intent-context-probes -- --branch master
cd intent-context-probes
python -m pip install -r requirements.txt
```

Use a short-lived repository-scoped GitHub token stored in Colab Secrets. Do
not place a token in the clone URL or notebook source. The provided Colab
driver implements this setup.

Run the environment check:

```bash
python scripts/00_check_environment.py --config configs/experiment.yaml
```

Then run the Qwen3-4B smoke test:

```bash
python scripts/03_run_model.py \
  --config configs/experiment.yaml \
  --model qwen3_4b \
  --smoke-test
```

The smoke test downloads `Qwen/Qwen3-4B`, renders one safe matched
conversation, extracts selected final-position hidden states, generates two
short responses, validates the saved artifact and copies it to the persistent
artifact root when configured.

## Expected smoke-test outputs

```text
artifacts/
├── environment.json
├── model_metadata/
│   └── qwen3_4b.json
└── smoke_tests/
    └── qwen3_4b/
        ├── smoke_test.json
        └── vectors.npz
```

`artifacts/smoke_tests/` is ignored by Git. The JSON metadata in
`artifacts/model_metadata/` may be committed after it is reviewed.

## Local verification without a GPU

The following tests exercise configuration, hashing, rendering contracts and
layer selection without importing or downloading the language model:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts
```

The environment checker is expected to report missing GPU/ML dependencies on
an ordinary CPU-only development machine. Use Colab for the required runtime
check.

## Milestone 2: author, audit and freeze the dataset

Use [`data/templates/scenario_template.json`](data/templates/scenario_template.json)
as an authoring aid. The template is not experimental data. Store one complete
JSON object per line in `data/raw/scenarios.jsonl`.

Draft in small batches and run validation after each batch:

```bash
python scripts/01_validate_dataset.py \
  --input data/raw/scenarios.jsonl \
  --audit-dir data/audits \
  --refresh-review-templates
```

`--refresh-review-templates` intentionally replaces entries in the two human
review sheets. Use it after changing the draft, before completing the review;
do not use it after entering review decisions unless those decisions are now
stale.

Inspect `data/audits/length_audit.csv` and `lexical_audit.csv`. Then complete:

- `pairwise_semantic_audit.csv`: enter `yes` for every satisfied check and add
  the reviewer name;
- `blinded_label_audit.csv`: assign `benign_evidence`,
  `suspicious_evidence`, or `ambiguous` without opening
  `blinded_label_key.csv` first;
- each scenario's `audit.manual_label_confirmed`: set it to `true` only after
  the scenario has actually been reviewed.

Rerun validation without replacing the completed review sheets:

```bash
python scripts/01_validate_dataset.py \
  --input data/raw/scenarios.jsonl \
  --audit-dir data/audits
```

Only after all 32 scenarios and both reviews are complete, freeze explicitly:

```bash
python scripts/01_validate_dataset.py \
  --input data/raw/scenarios.jsonl \
  --audit-dir data/audits \
  --freeze \
  --freeze-hash data/FROZEN_DATASET.sha256
```

The freeze command requires exactly eight scenarios in each domain and refuses
incomplete human audits. It rewrites the JSONL deterministically, records the
hash and manifest, and detects later byte changes. A genuine post-freeze
correction requires both `--allow-revision` and `--revision-note`; all later
artifacts must then be rerun.

Build the experimental prefix tables only after freezing:

```bash
python scripts/02_build_prefixes.py --config configs/experiment.yaml
```

For draft inspection only, use `--allow-draft`. The final model runs must use
the verified frozen dataset.

Expected Milestone 2 outputs:

```text
data/
├── FROZEN_DATASET.sha256
├── raw/scenarios.jsonl
├── processed/prefixes.csv
├── processed/prefixes.parquet
└── audits/
    ├── structural_audit.json
    ├── length_audit.csv
    ├── lexical_audit.csv
    ├── pairwise_semantic_audit.csv
    ├── blinded_label_audit.csv
    ├── blinded_label_key.csv
    └── freeze_manifest.json
```

After those outputs pass inspection, run both checkpoints sequentially, then
train TF-IDF baselines before activation probes and evaluate with nested
leave-one-domain-out validation.
