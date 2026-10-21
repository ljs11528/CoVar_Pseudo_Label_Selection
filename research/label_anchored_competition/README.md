# Label-anchored competitor verification

Research branch for the 2026-10-20 change of direction. This is a candidate mechanism, not an established improvement over CoVar or existing prototype methods.

- [Detailed experiment plan (Chinese)](docs/EXPERIMENT_PLAN.md)
- [Machine-readable frozen protocol](protocol.json)
- [Execution log](docs/EXECUTION_LOG.md)
- [Experiment ledger](records/experiments.json)

The first implementation keeps the classifier pseudo-label unchanged. It tests whether residual class probabilities aligned with label-supported prototype evidence add useful information beyond ordinary uncertainty, prototype disagreement, and probability–prototype fusion.

The first formal teacher must exclude the verifier images from backbone supervision and prototype construction. Existing full-label teachers are exploratory only. Full training comparisons follow the offline mechanism gate; they are not queued unconditionally.

As of 2026-10-21, the formal seed42 teacher and all three stages of A/B/C/D plus sampling sensitivity are complete. Teacher training took 15 hours 12 minutes; the subsequent diagnostic queue took 2 hours 25 minutes. The continuation gate did not pass, so seed43 and the end-to-end matrix were not started. See the [verified runtime and status](docs/P1_RUNTIME_AND_STATUS.md).

The 1,323-label VOC split remains fixed at 1,191 backbone/prototype images and 132 verifier-only images. Dataset checks, the four-device CUDA preflight and the two-update real-model smoke passed. Formal outputs are available; the current implementation did not establish the proposed mechanism's incremental benefit.

Run the numerical/statistical checks with:

```bash
python -m research.label_anchored_competition.run_selfcheck --output /new/selfcheck.json
```

For a completed formal checkpoint, use `export_frozen --help` followed by `analyze_frozen --help`. Exports enforce the fixed teacher, split, stage and protocol identities. Only D_rel fits heads and chooses fusion temperatures. Analyses report error-positive AP/AUROC, tie-aware risk–coverage, matched precision/coverage, calibration, weight mass and paired image intervals. Synthetic integration requires an explicit flag and cannot be mistaken for a formal cache.

`run_frozen_queue --help` describes the finite P1 execution queue. It waits for the preparation teacher, analyzes each fixed stage, and can run seed43 only after the predeclared resource gate passes. It never starts the G0–G7 matrix. `make_release` packages committed source only; `deploy_release` checks every file before making an immutable server release. `collect_status` retrieves small checksummed status/results without copying images, feature tensors or checkpoints to GitHub.

The queued export includes pinned native CSL/CoVar weights on the same full-image predictions and a deterministic 64-image full-resolution cache. The analysis reports 50 rankings and 24 fitted heads. The [experiment ledger](records/experiments.json) distinguishes running, queued and still pending work; full-image rendering and later interventions remain pending. A completed smoke or a live process is not evidence of research improvement.
