# Label-anchored competitor verification

Research branch for the 2026-10-20 change of direction. This is a candidate mechanism, not an established improvement over CoVar or existing prototype methods.

- [Detailed experiment plan (Chinese)](docs/EXPERIMENT_PLAN.md)
- [Machine-readable frozen protocol](protocol.json)
- [Execution log](docs/EXECUTION_LOG.md)
- [Experiment ledger](records/experiments.json)
- [Detailed P1 results, BN audit and next-step decision (Chinese)](docs/PHASE1_RESULTS_AND_AUDIT_20261023.md)
- [Bounded MC-preserving audit plan (Chinese)](docs/BOUNDED_SCORE_AUDIT_PLAN_20261023.md)
- [Fixed post-hoc scoring audit protocol](score_audit_protocol.json)
- [Completed MC-preserving audit, selection limitations and stopping decision (Chinese)](docs/BOUNDED_SCORE_AUDIT_RESULTS_20261023.md)
- [Bounded anchor-objection trust plan (Chinese)](docs/ANCHOR_TRUST_AUDIT_PLAN_20261023.md)
- [Fixed anchor-trust protocol](anchor_trust_protocol.json)

The first implementation keeps the classifier pseudo-label unchanged. It tests whether residual class probabilities aligned with label-supported prototype evidence add useful information beyond ordinary uncertainty, prototype disagreement, and probability–prototype fusion.

The first formal teacher must exclude the verifier images from backbone supervision and prototype construction. Existing full-label teachers are exploratory only. Full training comparisons follow the offline mechanism gate; they are not queued unconditionally.

As of 2026-10-21, the formal seed42 teacher and all three stages of A/B/C/D plus sampling sensitivity are complete. Teacher training took 15 hours 12 minutes; the subsequent diagnostic queue took 2 hours 25 minutes. The continuation gate did not pass, so seed43 and the end-to-end matrix were not started. See the [verified runtime and status](docs/P1_RUNTIME_AND_STATUS.md).

The 1,323-label VOC split remains fixed at 1,191 backbone/prototype images and 132 verifier-only images. Dataset checks, the four-device CUDA preflight and the two-update real-model smoke passed. Formal outputs are available; the current implementation did not establish the proposed mechanism's incremental benefit.

The 2026-10-23 audit found and repaired a deferred-BN-restoration checkpoint bug. All three old checkpoints were reevaluated on the complete validation set in FP32 and FP16. Reloaded FP16 mIoU differed from the training logs by +0.050, -0.063 and +0.064 points. Eight cached images per stage reproduced FP32 logits exactly. These checks preserve the old paired negative results; they do not prove that corrected BN snapshots would yield identical error rankings. The repaired saver has serialization regression tests and refuses to silently resume an old-policy checkpoint as a repaired run. No corrected full teacher or new-method training has been started.

The [summary data and figures](records/phase1_summary_20261023/) retain all 50 rankings at both sampling caps. Reproduce them with `summarize_phase1 --help`. Current evidence supports stopping expansion of the original evidence formula, with bounded MC-preserving probe and prototype-quality audits before another full teacher.

The bounded audit is now complete for all three existing-cache stages. It used image-nested D_rel CV, frozen MC offsets, an exact zero fallback, 500/2,000 updates, and two fixed minibatch seeds. The D_rel-only resource gate failed; no repaired teacher was started. Early/middle development gains did not establish stable incremental utility. Late final e models selected zero under all four budget/seed settings. The large pooled late OOF loss must not be interpreted as uniform within-fold harm: fold-specific score scales and selected forms matter. Read the [full report](docs/BOUNDED_SCORE_AUDIT_RESULTS_20261023.md) before interpreting the negative gate as absence of conditional information. Sparse e also reduced the registered additive basis to a linear one. All selections, curves, fixed strata, cache checks, and read-only model replays are retained in [audit records](records/bounded_score_audit_20261023/).

Run the numerical/statistical checks with:

```bash
python -m research.label_anchored_competition.run_selfcheck --output /new/selfcheck.json
```

For a completed formal checkpoint, use `export_frozen --help` followed by `analyze_frozen --help`. Exports enforce the fixed teacher, split, stage and protocol identities. Only D_rel fits heads and chooses fusion temperatures. Analyses report error-positive AP/AUROC, tie-aware risk–coverage, matched precision/coverage, calibration, weight mass and paired image intervals. Synthetic integration requires an explicit flag and cannot be mistaken for a formal cache.

`run_frozen_queue --help` describes the finite P1 execution queue. It waits for the preparation teacher, analyzes each fixed stage, and can run seed43 only after the predeclared resource gate passes. It never starts the G0–G7 matrix. `make_release` packages committed source only; `deploy_release` checks every file before making an immutable server release. `collect_status` retrieves small checksummed status/results without copying images, feature tensors or checkpoints to GitHub.

The queued export includes pinned native CSL/CoVar weights on the same full-image predictions and a deterministic 64-image full-resolution cache. The analysis reports 50 rankings and 24 fitted heads. The [experiment ledger](records/experiments.json) distinguishes running, queued and still pending work; full-image rendering and later interventions remain pending. A completed smoke or a live process is not evidence of research improvement.
