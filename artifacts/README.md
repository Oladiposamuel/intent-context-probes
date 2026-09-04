# Artifact storage policy

This directory contains runtime outputs. Large activation files, generated
responses, fitted probes and smoke-test vectors are ignored by Git and must be
copied to the persistent root named by `MATS_PERSISTENT_ARTIFACT_ROOT`.

Small reviewed metadata files may be committed:

- `environment.json`
- `model_metadata/*.json`

Never commit model weights, caches or credentials.
