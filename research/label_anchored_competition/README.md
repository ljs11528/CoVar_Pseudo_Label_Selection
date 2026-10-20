# Label-anchored competitor verification

Research branch for the 2026-10-20 change of direction. This is a candidate mechanism, not an established improvement over CoVar or existing prototype methods.

- [Detailed experiment plan (Chinese)](docs/EXPERIMENT_PLAN.md)
- [Machine-readable frozen protocol](protocol.json)
- [Execution log](docs/EXECUTION_LOG.md)
- [Experiment ledger](records/experiments.json)

The first implementation keeps the classifier pseudo-label unchanged. It tests whether residual class probabilities aligned with label-supported prototype evidence add useful information beyond ordinary uncertainty, prototype disagreement, and probability–prototype fusion.

The first formal teacher must exclude the verifier images from backbone supervision and prototype construction. Existing full-label teachers are exploratory only. Full training comparisons follow the offline mechanism gate; they are not queued unconditionally.

As of 2026-10-20, the 1,323-label VOC split is fixed at 1,191 backbone/prototype images and 132 verifier-only images. Dataset checks, the four-device CUDA preflight and the two-update real-model smoke passed. The fresh seed42 teacher is training on four RTX 4090 devices. There are no formal results for the new mechanism yet.

Run the numerical/statistical checks with:

```bash
python -m research.label_anchored_competition.run_selfcheck --output /new/selfcheck.json
```

For a completed formal checkpoint, use `export_frozen --help` followed by `analyze_frozen --help`. Exports enforce the fixed teacher, split, stage and protocol identities. Only D_rel fits heads and chooses fusion temperatures. Analyses report error-positive AP/AUROC, tie-aware risk–coverage, matched precision/coverage, calibration, weight mass and paired image intervals. Synthetic integration requires an explicit flag and cannot be mistaken for a formal cache.

`run_frozen_queue --help` describes the finite P1 execution queue. It waits for the preparation teacher, analyzes each fixed stage, and can run seed43 only after the predeclared resource gate passes. It never starts the G0–G7 matrix. `make_release` packages committed source only; `deploy_release` checks every file before making an immutable server release. `collect_status` retrieves small checksummed status/results without copying images, feature tensors or checkpoints to GitHub.

The [experiment ledger](records/experiments.json) distinguishes running, queued and still pending work, including the native legacy-selector export and full-image examples. A completed smoke or a live process is not evidence of research improvement.
