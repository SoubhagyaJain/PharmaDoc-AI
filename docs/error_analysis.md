# Failure analysis

This report combines regression failures found during implementation with measured evaluation examples. Aggregate run-specific counts and sample IDs are published in `results.md` and saved JSON summaries.

| Failure category | What can go wrong | Current response / next experiment |
|---|---|---|
| Confusing layouts | Column order, overlapping text layers, and OCR/native duplicates can misgroup fields | Preserve coordinates/routes; review flags; extend layout-specific regression fixtures |
| Blurry/low-resolution scans | Drug names, decimal points, units, and punctuation can change | Retain source quotes; measure CER/WER and field scores by degradation; do not silently correct text |
| Missing fields | A medication strength can tempt a model to invent an administered dose | Null missing values, required-field review, and explicit false-fill metrics |
| Ambiguous medication information | Abbreviations or similar drug spellings can be misinterpreted | Preserve original spelling; do not expand ambiguous frequencies; no medical correctness inference |
| Incorrect evidence role | A real prescriber quote can be assigned as the patient | Field-context checks reject obvious swaps; gold-label precision still measures residual mistakes |
| LLM latency/context | Verbose citations and multiple medications may exceed the local time/context budget | Bounded calls, one repair attempt, explicit failure; rules remain the default |
| Checkbox marks | A compact dot can look filled under a simple ink threshold | Interior geometry marks compact/sparse cases ambiguous; detection and state scores remain separate |
| Rotated evidence | Sorting boxes after inverse rotation scrambles reading order | Read in corrected coordinates, restore boxes afterward; regression tests cover this |
| Native decoding on Windows | Malformed PDF exception lifetimes can retain a filesystem handle | Decode size-bounded memory streams; test cleanup on failed uploads |
| Worker imports | `paddle.py` next to a standalone script can shadow the framework | Remove the worker directory from the import search path before importing Paddle |
| Repeated-value annotation | Global first-substring matches label the wrong medication | Restrict labels to their generated medication block; label repeated dose/frequency mentions consistently |

## Practical limits

The corpus contains a small vocabulary, consistent metadata labels, and only six layout families per category. Excellent synthetic scores are evidence of correctness on these cases, not a claim about real clinical documents. Handwriting, complex tables, stamps over text, arbitrary language, medication terminology normalization, and treatment safety are outside the validated scope.

The resource limit checks do not replace a hardened parser sandbox. The demo is localhost-only. Native rendering has pixel bounds and stage deadline checks but is not killed by a separate hard-timeout worker.

Field corrections do not acknowledge every page-level uncertainty. Reviewed exports can still require human review when a page failed or contained weak OCR. A future workflow should explicitly track page-level review acknowledgement before claiming review completion.
