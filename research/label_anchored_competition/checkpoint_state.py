"""Flush transient BN rollback state before Lightning captures a state_dict."""

BN_SNAPSHOT_POLICY = "pending BN rollback flushed before state_dict capture"


def require_flushed_checkpoint(metadata):
    """Do not silently continue an old teacher under the repaired policy."""
    if metadata.get("bn_snapshot_policy") != BN_SNAPSHOT_POLICY:
        raise ValueError("Checkpoint lacks the repaired BN snapshot policy; preserve the old run and use a new teacher output directory")


def flush_pending_bn_statistics(model):
    """Make the checkpoint match the next normal forward's running statistics.

    The upstream model defers restoring running_mean/running_var until the next
    forward. Its Python rollback list is not in state_dict. Call this before
    trainer.save_checkpoint, rather than inside on_save_checkpoint after the
    state_dict may already have been captured. No parameters or RNG are changed.
    """
    count = len(getattr(model, "saved_bn_stats", ()))
    if count:
        model.restore_bn_stats()
        if getattr(model, "saved_bn_stats", ()):
            raise RuntimeError("Pending BN rollback was not cleared before checkpoint")
    return count
