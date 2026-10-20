# Label-anchored competitor verification

Research branch for the 2026-10-20 change of direction. This is a candidate mechanism, not an established improvement over CoVar or existing prototype methods.

- [Detailed experiment plan (Chinese)](docs/EXPERIMENT_PLAN.md)
- [Machine-readable frozen protocol](protocol.json)
- [Execution log](docs/EXECUTION_LOG.md)
- [Experiment ledger](records/experiments.json)

The first implementation keeps the classifier pseudo-label unchanged. It tests whether residual class probabilities aligned with label-supported prototype evidence add useful information beyond ordinary uncertainty, prototype disagreement, and probability–prototype fusion.

The first formal teacher must exclude the verifier images from backbone supervision and prototype construction. Existing full-label teachers are exploratory only. Full training comparisons follow the offline mechanism gate; they are not queued unconditionally.
