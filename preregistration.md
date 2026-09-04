# Preregistration draft

**Status:** DRAFT — not yet frozen or timestamped.

Samuel must review this document and commit the final version before inspecting
held-out test results. Editing this draft after test inspection would not count
as preregistration. Any later deviation must be recorded in
`logs/deviations.md`.

## Research questions

1. Does `Qwen/Qwen3-4B` linearly encode observable conversational evidence
   that contradicts a declared benign purpose?
2. Does full conversation history improve separation relative to the current
   user message alone, especially when the paired Turn-4 message is identical?
3. Does additional SafeRL post-training improve intent-evidence separation,
   increase broad topic-risk sensitivity, or both?
4. Do activation probes add predictive value beyond TF-IDF, input length and a
   directly prompted intent judge?
5. Does an internal signal precede, coincide with or follow cautious observable
   behaviour?

## Primary hypothesis

On held-out domains and paired Turn-4 examples, the full-history activation
probe will achieve higher domain-macro AUROC than the current-message-only
activation probe for `Qwen/Qwen3-4B`.

## Secondary hypotheses

- SafeRL will make suspicious evidence more linearly separable than the general
  chat checkpoint.
- A useful intent-evidence representation will retain low scores on matched
  benign branches rather than merely responding to dangerous subject matter.
- Full-history activations may outperform full-history TF-IDF, but this is not
  assumed.
- Probe scores will correlate with response caution, although a knowing-doing
  gap may remain.
- Shared underdetermined prefixes should not systematically separate future
  benign and suspicious branches.

## Fixed core design

- Models: `Qwen/Qwen3-4B` and `Qwen/Qwen3-4B-SafeRL`.
- Dataset: 32 matched four-turn scenarios across four domains.
- Turns 1 and 2 are shared within each pair.
- Turn 3 contains branch-specific observable evidence.
- Turn 4 is identical within each pair.
- Readout: final input position after the assistant generation marker.
- Layers: approximately 25%, 50%, 75% and 100% model depth.
- Probe: L2-regularized logistic regression.
- Outer evaluation: leave one risk domain out.
- Inner selection: leave one remaining training domain out.
- Fitting turns: 3 and 4.
- Primary evaluation: Turn 4.
- Primary metric: macro-domain AUROC.
- At most one result-driven follow-up, clearly labelled exploratory.

## Required baselines

- Full-history TF-IDF with logistic regression.
- Current-message TF-IDF with logistic regression.
- Full-history prompted intent judgement.
- Current-message prompted intent judgement.
- Input-length control.
- Permuted-label sanity check.

## Claims this experiment cannot support

- Access to a user's private or true intent.
- Causal use of a representation by the model.
- Generalization beyond the tested models and domains.
- Deployment readiness of an intent detector.
- Safety of an adaptive-disclosure system.

## Freeze checklist

- [ ] Dataset construction and labels reviewed manually.
- [ ] Split rules and metrics reviewed.
- [ ] Smoke test passed.
- [ ] No held-out test result inspected.
- [ ] UTC freeze timestamp added below.
- [ ] Final preregistration committed to Git.

**Freeze timestamp (UTC):** _not yet frozen_
