"""Train the formal D_sup-only MC teacher at fixed optimizer-step stages.

The upstream SSL losses and augmentation are retained. This wrapper registers
the previously manual LR scheduler, fixes single-device validation aggregation,
excludes verifier images from every backbone input stream, and saves fixed steps.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from .prepare_data import digest, disjoint_partitions, image_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--prepared-partition", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--pretrained", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    os.environ["PYTHONPATH"] = str(repo)+os.pathsep+os.environ.get("PYTHONPATH", "")
    for key in ("upstream", "prepared_partition", "data_root", "pretrained", "output", "protocol"):
        setattr(args, key, getattr(args, key).resolve())
    protocol = json.loads(args.protocol.read_text())
    spec = protocol["teacher"]
    if args.seed not in (spec["seed"], spec["confirmation_seed"]):
        parser.error("Only the preregistered preparation teacher seeds are allowed")
    if args.devices < 1 or spec["global_batch_size"] % args.devices:
        parser.error("Global batch must be divisible by the device count")
    audit = json.loads((args.prepared_partition / "COMPLETE.json").read_text())
    splits = args.prepared_partition / "splits"
    paths = dict(D_sup=splits / "1_8/labeled.txt", D_rel=splits / "rel.txt",
                 D_U=splits / "1_8/unlabeled.txt", validation=splits / "val.txt")
    role_ids = {}
    for role, path in paths.items():
        expected = audit["outputs"][str(path.relative_to(args.prepared_partition))]["sha256"]
        if digest(path) != expected:
            raise ValueError(f"Prepared split was modified: {role}")
        role_ids[role] = [image_identity(line) for line in path.read_text().splitlines()]
        if len(role_ids[role]) != audit["counts"][role]:
            raise ValueError("Prepared split count differs")
    disjoint_partitions(role_ids)
    if len(role_ids["D_sup"]) != protocol["partition"]["supervised_images"]:
        raise ValueError("Teacher is not restricted to the locked supervised budget")
    for role in ("D_sup", "D_U", "validation"):
        for line in paths[role].read_text().splitlines():
            if any(not (args.data_root / name).is_file() for name in line.split()):
                raise FileNotFoundError(f"Training/validation input absent: {role}")
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((args.output / "checkpoints").glob("*.ckpt")):
        raise FileExistsError("Existing teacher checkpoints require explicit --resume")
    sys.path.insert(0, str(args.upstream))
    os.chdir(args.upstream)
    import numpy as np
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger
    import yaml
    from dataset.semi import SemiDataset
    from model.model_helper import ModelBuilder
    from train.semi_supervised_train import SemiModule
    from util.loggingCallback import LoggingCallback

    pl.seed_everything(args.seed, workers=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    config_path = args.upstream / "configs/pascal.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    cfg["model"]["backbone"]["kwargs"]["pretrained"] = False
    cfg["dataset"]["root"] = str(args.data_root)
    model = ModelBuilder(cfg["model"])
    pretrained = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    incompatible = model.backbone.load_state_dict(pretrained, strict=False)
    missing_trainable = set(incompatible.missing_keys) & set(dict(model.backbone.named_parameters()))
    if missing_trainable:
        raise ValueError(f"Pretrained backbone has missing trainable parameters: {sorted(missing_trainable)}")
    total_steps = spec["total_optimizer_steps"]
    unlabeled = SemiDataset(**dict(cfg["dataset"], mode="train_u", id_path=str(paths["D_U"])))
    labeled = SemiDataset(**dict(cfg["dataset"], mode="train_l", id_path=str(paths["D_sup"]), nsample=len(unlabeled)))
    validation = SemiDataset(cfg["dataset"]["name"], str(args.data_root), "val", str(paths["validation"]))

    class EqualLengthUniqueValidation(torch.utils.data.Dataset):
        def __len__(self):
            return math.ceil(len(validation)/args.devices)

        def __getitem__(self, index):
            global_index = index*args.devices+rank
            image, mask, identity = validation[global_index % len(validation)]
            if global_index >= len(validation):
                mask = torch.full_like(mask, 255)
            return image, mask, identity

    per_device = spec["global_batch_size"]//args.devices
    def train_loader(dataset, stream):
        sampler = (torch.utils.data.DistributedSampler(dataset, num_replicas=args.devices,
                   rank=rank, shuffle=True, seed=args.seed+stream) if args.devices > 1 else None)
        return torch.utils.data.DataLoader(dataset, batch_size=per_device, sampler=sampler,
                   shuffle=sampler is None, drop_last=True, num_workers=args.workers,
                   pin_memory=True, persistent_workers=args.workers > 0)
    loaders = dict(labeled=train_loader(labeled, 0), unlabeled=train_loader(unlabeled, 1), mixed=train_loader(unlabeled, 2))
    val_loader = torch.utils.data.DataLoader(EqualLengthUniqueValidation(), batch_size=1,
                     shuffle=False, num_workers=args.workers, pin_memory=True)
    batch_iters = len(unlabeled)//spec["global_batch_size"]
    if batch_iters != spec["steps_per_epoch_reference"] or len(loaders["unlabeled"]) != batch_iters:
        raise ValueError("Device/sampler adaptation changed the fixed iteration budget")

    class Teacher(SemiModule):
        def configure_optimizers(self):
            super().configure_optimizers()
            return dict(optimizer=self.optimizer, lr_scheduler=dict(scheduler=self.scheduler, interval="step", frequency=1))

        def on_train_batch_end(self, outputs, batch, batch_idx):
            # Lightning owns the registered scheduler; do not run the upstream
            # manual scheduler a second time. Its complete state is checkpointed.
            pass

        def on_train_epoch_start(self):
            super().on_train_epoch_start()
            for loader in loaders.values():
                if hasattr(loader.sampler, "set_epoch"):
                    loader.sampler.set_epoch(self.current_epoch)

        def on_validation_epoch_end(self):
            intersection = self.all_gather(self.intersection_meter.sum).reshape(-1, self.nclass).sum(0)
            union = self.all_gather(self.union_meter.sum).reshape(-1, self.nclass).sum(0)
            per_class = intersection/(union+1e-10)*100
            self.log("val_mIOU", per_class.mean(), rank_zero_only=True, sync_dist=True, batch_size=1)
            self.logging_callback.info(f"validation global_step={self.global_step} mIoU={per_class.mean():.6f}")

        def configure_callbacks(self):
            self.logging_callback = LoggingCallback(log_file=str(args.output / "output.log"), monitor="val_mIOU", mode="max")
            return [self.logging_callback]

        def on_save_checkpoint(self, checkpoint):
            checkpoint["label_anchored_study"] = dict(
                protocol_sha256=digest(args.protocol), partition_receipt_sha256=digest(args.prepared_partition / "COMPLETE.json"),
                role_counts=audit["counts"], formal_supervision="D_sup only", seed=args.seed,
                source_identity=source_identity,
                rng_by_rank=getattr(self, "_rng_by_rank", None),
                worker_limitation="Original random augmentation workers are not serialized; resumption is not claimed bitwise identical.")

        def on_load_checkpoint(self, checkpoint):
            stored = checkpoint.get("label_anchored_study", {})
            if stored.get("protocol_sha256") != digest(args.protocol) or stored.get("partition_receipt_sha256") != digest(args.prepared_partition / "COMPLETE.json"):
                raise ValueError("Resume checkpoint belongs to another protocol/data partition")
            self._restored_rng = stored.get("rng_by_rank")

        def on_train_start(self):
            states = getattr(self, "_restored_rng", None)
            if states:
                state = states[self.global_rank]
                random.setstate(state["python"])
                np.random.set_state(state["numpy"])
                torch.set_rng_state(state["torch"].cpu())
                torch.cuda.set_rng_state(state["cuda"].cpu(), self.device)

    stage_steps = set(spec["stage_steps"].values())
    source_file = repo / "source_manifest.json"
    source_identity = json.loads(source_file.read_text()) if source_file.exists() else dict(
        training_tool_sha256=digest(__file__), note="source bundle manifest absent; local development only")

    class StepCheckpoints(pl.Callback):
        def on_train_start(self, trainer, module):
            self.started_wall = time.monotonic()
            self.last_wall = self.started_wall
            self.last_step = trainer.global_step

        def state_dict(self):
            return dict(saved_steps=sorted(getattr(self, "saved_steps", set())))

        def load_state_dict(self, state_dict):
            self.saved_steps = set(state_dict["saved_steps"])

        def save(self, trainer, module, name):
            local = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                         cuda=torch.cuda.get_rng_state(module.device))
            if torch.distributed.is_initialized():
                gathered = [None]*trainer.world_size
                torch.distributed.all_gather_object(gathered, local)
            else:
                gathered = [local]
            module._rng_by_rank = gathered
            directory = args.output / "checkpoints"
            directory.mkdir(exist_ok=True)
            final = directory / name
            partial = final.with_suffix(".partial.ckpt")
            trainer.save_checkpoint(str(partial))
            if trainer.is_global_zero:
                partial.replace(final)
            trainer.strategy.barrier()

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            step = trainer.global_step
            if trainer.is_global_zero and (step <= 30 or step % 100 == 0):
                torch.cuda.synchronize(module.device)
                now = time.monotonic()
                progress = dict(state="running", global_step=step, target_steps=2 if args.smoke else total_steps,
                                elapsed_wall_seconds=now-self.started_wall,
                                seconds_per_step_recent=(now-self.last_wall)/max(1, step-self.last_step),
                                lr=[group["lr"] for group in module.optimizer.param_groups],
                                scheduler_last_epoch=module.scheduler.last_epoch,
                                peak_allocated_mib=torch.cuda.max_memory_allocated(module.device)/2**20,
                                updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
                temporary = args.output / "progress.partial.json"
                temporary.write_text(json.dumps(progress, indent=2)+"\n")
                temporary.replace(args.output / "progress.json")
                self.last_wall, self.last_step = now, step
            if args.smoke:
                return
            saved = getattr(self, "saved_steps", set())
            if step in stage_steps and step not in saved:
                self.saved_steps = saved | {step}
                self.save(trainer, module, f"step_{step:06d}.ckpt")
            if step and (step % batch_iters == 0 or step == total_steps):
                self.save(trainer, module, "last.ckpt")

    module = Teacher(**dict(cfg["train"], model=model, save_path=str(args.output),
                           batch_iters=batch_iters, total_iters=total_steps,
                           threshold_strategy="fixed", fixed_threshold=spec["mc_threshold"],
                           enable_visual_artifacts=False, monitor_thresholds=[]))
    run_signature = dict(protocol_sha256=digest(args.protocol), partition_receipt_sha256=digest(args.prepared_partition / "COMPLETE.json"),
                         seed=args.seed, devices=args.devices, global_batch_size=spec["global_batch_size"],
                         total_steps=total_steps, stage_steps=spec["stage_steps"], smoke=args.smoke,
                         backbone_pretrained_sha256=digest(args.pretrained), config_sha256=digest(config_path),
                         source_identity=source_identity)
    if rank == 0:
        manifest = dict(signature=run_signature, role_counts=audit["counts"], started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                        initialization_missing_keys=incompatible.missing_keys, initialization_unexpected_keys=incompatible.unexpected_keys,
                        torch=torch.__version__, lightning=pl.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                        gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                        workers=args.workers, validation="equal length ranks, padded rows ignored; each validation image counted once",
                        LR_scheduler="registered with Lightning, one step per optimizer update, checkpoint restored",
                        resume=str(args.resume) if args.resume else None,
                        worker_resume="RNG saved per rank; random augmentation worker states not serialized, no bitwise claim")
        existing = args.output / "run_manifest.json"
        if existing.exists() and json.loads(existing.read_text())["signature"] != run_signature:
            raise ValueError("Existing run manifest differs")
        existing.write_text(json.dumps(manifest, indent=2)+"\n")
    callback = StepCheckpoints()
    trainer = pl.Trainer(accelerator="gpu", devices=args.devices,
                         strategy="ddp" if args.devices > 1 else "auto",
                         max_epochs=-1, max_steps=2 if args.smoke else total_steps,
                         precision=spec["precision"], sync_batchnorm=args.devices > 1,
                         logger=CSVLogger(str(args.output), name="csv"), callbacks=[callback],
                         enable_checkpointing=False, use_distributed_sampler=False,
                         num_sanity_val_steps=0, enable_progress_bar=False,
                         log_every_n_steps=1 if args.smoke else 25,
                         limit_train_batches=2 if args.smoke else 1.0,
                         limit_val_batches=2 if args.smoke else 1.0,
                         deterministic="warn", benchmark=False)
    trainer.fit(module, train_dataloaders=loaders, val_dataloaders=val_loader,
                ckpt_path=str(args.resume) if args.resume else None)
    if args.smoke:
        callback.save(trainer, module, "smoke_only.ckpt")
    if trainer.is_global_zero:
        expected = 2 if args.smoke else total_steps
        if trainer.global_step != expected:
            raise ValueError("Training exited without reaching its fixed update budget")
        final = dict(state="complete", global_step=trainer.global_step, smoke=args.smoke,
                     completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), signature=run_signature,
                     checkpoint_sha256={p.name:digest(p) for p in sorted((args.output / "checkpoints").glob("*.ckpt"))},
                     optimizer_state_nonempty=bool(module.optimizer.state), scheduler_last_epoch=module.scheduler.last_epoch,
                     scope="synthetic/real-data smoke only" if args.smoke else "formal D_sup-only MC teacher, not new-method performance")
        (args.output / "COMPLETE.json").write_text(json.dumps(final, indent=2)+"\n")
        print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
