# Revision package validation — 2026-09-28

- Experiment and recovery suite: 42 tests passed (`python -m unittest discover -s reviewer_experiments/tests -v` at that stage).
- Delivery preset and summary tests added subsequently: 3 tests passed (`python -m unittest reviewer_experiments.tests.test_revision_delivery -v`). Total: 45 passing tests. No experiment code changed after these checks.
- Exported all eight MAT cubes from the supplied recovery ZIP; validated finite HWC tensors and monotonic nominal wavelength axes. Original MAT hashes are retained in the local export audit.
- The controls preset prints five models × one task × one seed × five folds = 25 runs. Printing the plan does not start training.
- Reviewed the rendered manuscript page by page. The final render has 31 pages, 18 tables, seven figures and nine editable Office Math objects (seven replaced equations plus two retained inline expressions). Unchanged pages were checked against the preceding inspected render.
- Verified that obsolete bracketed completion flags and tracked insertions/deletions are absent. The inherited comment part is empty. Author declarations remain explicitly subject to confirmation.
- Archived-prediction verification covered 630 prediction files, 126 complete model/task/seed groups and 252 reference BA/AUROC cells; see `evidence/prediction_verification.json`.

Final DOCX SHA-256:

```text
9c35d18aa12f9b894908bcf2cf77233610a356989e59e040bb3d7a557e3635b0
```

Validation limits: no full historical PT bundles were available locally, so a real-data merge and new GPU training were not executed. Synthetic tests cover merge rejection, overlap deduplication, preset budgets, incomplete folds, differing protocols and single-seed uncertainty. The manuscript does not report newly completed controlled ablations or newly recovered-cohort results. This is an author-review draft, not evidence that all reviewer requests have been satisfied.
