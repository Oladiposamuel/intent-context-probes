# Intent or Topic Risk?

This repository implements the MATS application experiment described in
[`MATS_INTENT_PROBE_PROJECT_BRAIN.md`](MATS_INTENT_PROBE_PROJECT_BRAIN.md).
It tests whether a general Qwen3 chat model and its SafeRL counterpart linearly
encode observable evidence of harmful intent across matched multi-turn
conversations.

## Current milestone

Model and dataset preparation is complete:

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

The reviewed 32-scenario dataset is frozen. It contains 256 prefix rows, 64
primary Turn-4 rows and 128 supervised Turn-3/Turn-4 rows. Bulk activation
extraction is implemented with rendered-input deduplication, per-domain atomic
checkpoints, safe resumption, hash validation and paired-vector assertions.
Response generation, prompted judgements, baselines, probe training and
evaluation remain later milestones.

Turn-4 response generation and the fixed prompted-judgement baseline are now
available through `scripts/03_run_model.py`. They use deterministic decoding,
checkpoint after every domain, validate model/dataset identity on resume, and
copy outputs to `MATS_PERSISTENT_ARTIFACT_ROOT`. Run both operations together
for one checkpoint:

```bash
python scripts/03_run_model.py --config configs/experiment.yaml \
  --model qwen3_4b --generate-turn4-responses --run-prompted-judge
```

Repeat with `--model qwen3_4b_saferl`. Generated artifacts remain ignored by
Git and must not be committed.

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

Both model revisions are pinned in `configs/experiment.yaml`. The SafeRL smoke
test must pass before its bulk extraction:

```bash
python scripts/03_run_model.py \
  --config configs/experiment.yaml \
  --model qwen3_4b_saferl \
  --smoke-test
```

After inspecting the smoke-test JSON and confirming `status: passed`, run:

```bash
python scripts/03_run_model.py \
  --config configs/experiment.yaml \
  --model qwen3_4b_saferl \
  --extract-activations
```

Repeat the extraction command with `--model qwen3_4b` for the general
checkpoint. Each run writes `full_history.npz` and `current_message.npz` under
`artifacts/activations/<model-alias>/`, with shape `[256, 4, hidden_size]`, plus
JSON metadata. If a runtime disconnects after a domain checkpoint, rerun the
same command; compatible completed rows are validated and reused.

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

The blinded audit is the preferred protocol. If it is deliberately waived,
the strict default must be overridden with both an explicit flag and written
reason. The pairwise semantic audit remains mandatory:

~~~bash
python scripts/01_validate_dataset.py \
  --input data/raw/scenarios.jsonl \
  --audit-dir data/audits \
  --freeze \
  --skip-blinded-audit \
  --blinded-audit-waiver-note "Explain the protocol deviation here."
~~~

The waiver and its consequence are recorded in the freeze manifest. See
[the protocol-deviation log](docs/PROTOCOL_DEVIATIONS.md). Do not report a
waived audit as completed or independently validated.

Rerun validation without replacing the completed review sheets:

```bash
python scripts/01_validate_dataset.py \
  --input data/raw/scenarios.jsonl \
  --audit-dir data/audits
```

Under the preferred protocol, freeze only after all 32 scenarios and both
reviews are complete:

```bash
python scripts/01_validate_dataset.py \
  --input data/raw/scenarios.jsonl \
  --audit-dir data/audits \
  --freeze \
  --freeze-hash data/FROZEN_DATASET.sha256
```

The freeze command requires exactly eight scenarios in each domain and refuses
an incomplete pairwise review. By default it also refuses an incomplete
blinded audit; the documented waiver above is the only exception. It rewrites
the JSONL deterministically, records the hash and manifest, and detects later
byte changes. A genuine post-freeze correction requires both
`--allow-revision` and `--revision-note`; all later artifacts must then be
rerun.

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


## CPU analysis and early-prefix sanity

After the activation, baseline and probe artifacts are complete, run the verified
statistics and H6 trajectory checks on CPU:

```bash
python scripts/07_evaluate.py
python scripts/08_analyze_trajectory.py
```

The trajectory runner refits each outer-fold probe using only registered Turns
3-4 from the three training domains and the already selected layer and
regularization. It then scores Turns 1-4 in the held-out domain. The runner
refuses to complete unless the shared benign/suspicious inputs at Turns 1-2
receive equal scores and yield chance branch AUROC. Generated trajectory CSV
and JSON files remain ignored by Git and are synchronized to persistent
storage when configured.


## Prompted-judgement evaluation

Evaluate the four saved prompted-judgement artifacts without loading either
language model:

```bash
python scripts/09_evaluate_prompted_judgements.py
```

The evaluator validates all 64 expected IDs, frozen dataset hashes, checkpoint
revisions and context modes. It reports available-case AUROC and separate
complete-pair metrics. A failed parse remains missing: it is never regenerated,
repaired, or imputed during evaluation. Paired bootstrap intervals use only
complete scenario pairs and preserve domain stratification. Generated score
tables and metrics remain ignored by Git and are copied to persistent storage.


## Blinded AI-assisted response annotation

Prepare the randomized annotation package only after all 128 response artifacts
are present:

```bash
python scripts/10_prepare_response_annotations.py
```

This creates separate 128-row primary and 26-row repeat sheets plus private
identity and repeat-link keys under `artifacts/annotations/`. Share only the
two files beginning with `blinded_` with the annotator. Follow
`docs/AI_ASSISTED_ANNOTATION_RUBRIC.md` and do not inspect private keys before
both passes are complete.

Save completed sheets as `completed_primary.csv` and
`completed_repeat.csv`, then validate, hash-lock, unblind, measure repeat
agreement, and compute descriptive behaviour metrics:

```bash
python scripts/11_finalize_response_annotations.py
```

All sheets, keys, completed ratings and analysis outputs are Git-ignored and
synchronized to persistent storage. Report the annotator as a blinded
language-model annotator, never as a human.
