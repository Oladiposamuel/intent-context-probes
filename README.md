# Intent or Topic Risk?

This repository implements the MATS application experiment described in
[`MATS_INTENT_PROBE_PROJECT_BRAIN.md`](MATS_INTENT_PROBE_PROJECT_BRAIN.md).
It tests whether a general Qwen3 chat model and its SafeRL counterpart linearly
encode observable evidence of harmful intent across matched multi-turn
conversations.

## Current milestone

Milestone 1 is implemented:

- reproducible experiment configuration;
- Colab/GPU environment validation;
- Qwen model revision and metadata capture;
- shared chat-template rendering;
- assistant-decision-position activation extraction;
- a safe, built-in Qwen3-4B smoke test;
- local unit tests that do not download a model;
- a thin Colab driver notebook.

Dataset construction, bulk extraction, response annotation, text baselines,
probe training and evaluation are intentionally not implemented yet. They
should be built only after the smoke test passes.

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

## Next milestone after the smoke test

1. Construct and manually review 32 matched scenarios.
2. Validate and freeze `data/raw/scenarios.jsonl`.
3. Build full-history and current-message prefixes.
4. Run both checkpoints sequentially.
5. Train TF-IDF baselines before activation probes.
6. Evaluate with nested leave-one-domain-out validation.

Do not proceed to those steps until the smoke-test output has been inspected.
