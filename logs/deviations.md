# Deviations from the frozen plan

No deviations recorded. Add entries only after the preregistration is frozen.

Each entry must state the timestamp, original plan, change, reason, whether the
change was made before or after observing the affected result, and the expected
effect on interpretation.

## 2026-09-04 — Response annotation postponed

- Original plan: annotate generated responses before training baselines and probes.
- Change: run TF-IDF, length controls and activation probes first; retain response artifacts for later blinded annotation.
- Reason: prioritize the primary representation result and avoid 60–90 minutes of manual work before it is needed.
- Timing: decided before viewing any baseline or probe result.
- Expected effect: none on predictive results; RQ5 remains incomplete until annotation resumes.

## 2026-09-04 — Length-control implementation corrected

- Original plan: use current/full word counts, tokenizer token counts, and turn index.
- Initial implementation: mistakenly substituted character counts for tokenizer token counts.
- Correction: read the exact Qwen rendered-input token counts saved with the activation artifacts and rerun the registered control.
- Timing: identified immediately after the first baseline output and before uncertainty estimates or conclusions were finalized.
- Expected effect: only the length-control estimate may change; TF-IDF and activation predictions are unaffected.

## 2026-09-04 — Model execution order reversed

- Original plan: run the base model before the safety-tuned model.
- Change: complete the SafeRL extraction before the base-model extraction.
- Reason: the SafeRL pipeline was ready first after smoke testing.
- Timing: execution-order change occurred before probe training or viewing probe results.
- Expected effect: none; revisions, data, rendering, layers, and evaluation are frozen and each model is processed independently.
