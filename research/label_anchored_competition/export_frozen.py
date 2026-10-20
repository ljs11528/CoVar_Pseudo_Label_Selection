"""Export one formal frozen teacher: D_sup prototypes, D_rel and evaluation caches.

Validation GT is read only for sample validity, correctness and diagnostic regions.
It never enters the prototype bank or the verifier fitting pool.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .core import PrototypeAccumulator, sample_decoder_features, score_prediction
from .prepare_data import digest, image_identity, source_revision


def fixed_rng(namespace, identity, seed=0):
    number = int.from_bytes(hashlib.sha256(f"{namespace}|{seed}|{identity}".encode()).digest()[:8], "little")
    return np.random.default_rng(number)


def boundary_pixels(mask, distance=3):
    valid = mask != 255
    output = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    for dy in range(-distance, distance+1):
        for dx in range(-distance, distance+1):
            if not 0 < abs(dx)+abs(dy) <= distance:
                continue
            y0, y1 = max(0, -dy), min(h, h-dy)
            x0, x1 = max(0, -dx), min(w, w-dx)
            center = mask[y0:y1, x0:x1]
            other = mask[y0+dy:y1+dy, x0+dx:x1+dx]
            output[y0:y1, x0:x1] |= (center != other) & valid[y0:y1, x0:x1] & valid[y0+dy:y1+dy, x0+dx:x1+dx]
    return output


def atomic_json(path, value):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prepared-partition", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=["early", "middle", "late"], required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for name in ("upstream", "checkpoint", "prepared_partition", "data_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    package = Path(__file__).resolve().parent
    protocol = json.loads((package / "protocol.json").read_text())
    checkpoint_hash = digest(args.checkpoint)
    partition_hash = digest(args.prepared_partition / "COMPLETE.json")
    source = source_revision(package)
    signature = dict(source_commit=source, exporter_sha256=digest(__file__),
                     protocol_sha256=digest(package / "protocol.json"), checkpoint_sha256=checkpoint_hash,
                     partition_receipt_sha256=partition_hash, stage=args.stage,
                     feature_layer=protocol["teacher"]["feature_layer"], seed=None)
    if (args.output / "COMPLETE.json").exists():
        existing = json.loads((args.output / "COMPLETE.json").read_text())
        if existing["signature"] != signature:
            # The seed is stored in the checkpoint, checked below before reuse.
            if {k:v for k,v in existing["signature"].items() if k != "seed"} != {k:v for k,v in signature.items() if k != "seed"}:
                raise ValueError("Existing frozen export signature differs")
        print("Existing completed frozen export preserved", flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    study = checkpoint.get("label_anchored_study", {})
    if (study.get("formal_supervision") != "D_sup only" or
            study.get("protocol_sha256") != signature["protocol_sha256"] or
            study.get("partition_receipt_sha256") != partition_hash or
            checkpoint["global_step"] != protocol["teacher"]["stage_steps"][args.stage]):
        raise ValueError("Checkpoint is not the preregistered D_sup-only formal teacher")
    signature["seed"] = study["seed"]
    split_root = args.prepared_partition / "splits"
    partitions = json.loads((args.prepared_partition / "partitions.json").read_text())
    supplied = dict(D_sup=split_root / "1_8/labeled.txt", D_rel=split_root / "rel.txt", validation=split_root / "val.txt")
    for role in ("D_sup", "D_rel"):
        actual = {image_identity(line) for line in supplied[role].read_text().splitlines()}
        if actual != set(partitions[role]):
            raise ValueError("Export data roles differ from fixed partition")
    sys.path.insert(0, str(args.upstream))
    import yaml
    from model.model_helper import ModelBuilder
    from dataset.semi import SemiDataset
    cfg = yaml.safe_load((args.upstream / "configs/pascal.yaml").read_text())
    cfg["model"]["backbone"]["kwargs"]["pretrained"] = False
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = ModelBuilder(cfg["model"])
    model.load_state_dict({name.removeprefix("model."):value for name, value in checkpoint["state_dict"].items() if name.startswith("model.")}, strict=True)
    del checkpoint
    model = model.to(args.device).eval()
    captured = {}
    def capture(module, inputs):
        captured["features"] = inputs[0].detach()
    hook = model.decoder.classifier.register_forward_pre_hook(capture)
    datasets = {role:SemiDataset("pascal", str(args.data_root), "val", str(path)) for role, path in supplied.items()}
    prototype_file = args.output / "prototypes.npz"
    prototype_meta = args.output / "prototype_manifest.json"
    with torch.no_grad():
        if prototype_file.exists() and prototype_meta.exists():
            stored = json.loads(prototype_meta.read_text())
            if stored["signature"] != signature or stored["prototype_sha256"] != digest(prototype_file):
                raise ValueError("Existing prototype identity differs")
            with np.load(prototype_file, allow_pickle=False) as archive:
                prototypes = torch.from_numpy(archive["prototypes"].copy()).to(args.device)
        else:
            bank = PrototypeAccumulator(protocol["dataset"]["classes"], 256, partitions["D_sup"])
            image_means, image_counts, identities = [], [], []
            data = datasets["D_sup"]
            for index in range(len(data)):
                image, mask, line = data[index]
                identity = image_identity(line)
                mask_array = mask.numpy()
                h, w = mask_array.shape
                model(image[None].to(args.device), False)
                fmap = captured["features"]
                chosen, labels = [], []
                for c in range(protocol["dataset"]["classes"]):
                    valid = np.flatnonzero(mask_array.ravel() == c)
                    if not len(valid):
                        continue
                    cap = min(protocol["prototypes"]["sample_per_image_class"], len(valid))
                    selection = fixed_rng(f"prototype_class_{c}", identity).choice(valid, cap, replace=False)
                    chosen.append(selection)
                    labels.append(np.full(cap, c, dtype=np.int64))
                if not chosen:
                    raise ValueError("Supervised image has no valid class support")
                indices = np.concatenate(chosen)
                class_labels = torch.from_numpy(np.concatenate(labels))
                yx = torch.from_numpy(np.stack((indices//w, indices % w), 1))
                features = sample_decoder_features(fmap, yx, (h, w)).cpu()
                before_sums, before_count = bank.sums.clone(), bank.image_counts.clone()
                bank.add_image(identity, features, class_labels)
                image_means.append((bank.sums-before_sums).float().numpy())
                image_counts.append((bank.image_counts-before_count).numpy())
                identities.append(identity)
                if (index+1) % 25 == 0:
                    atomic_json(args.output / "export_status.json", dict(state="running", phase="D_sup_prototypes", images=index+1,
                                target=len(data), signature=signature, updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
            prototypes = bank.finalize().to(args.device)
            temporary = prototype_file.with_suffix(".partial.npz")
            np.savez_compressed(temporary, prototypes=prototypes.cpu().numpy(),
                                image_counts=bank.image_counts.numpy(), sampled_pixel_counts=bank.sampled_pixel_counts.numpy(),
                                image_ids=np.asarray(identities), per_image_class_means=np.stack(image_means),
                                per_image_class_present=np.stack(image_counts))
            temporary.replace(prototype_file)
            atomic_json(prototype_meta, dict(signature=signature, prototype_sha256=digest(prototype_file),
                        source="D_sup GT only", independent_support_images=bank.image_counts.tolist(),
                        estimator=protocol["prototypes"]["frozen_estimator"]))
        prototype_hash = digest(prototype_file)
        all_records = []
        for role in ("D_rel", "validation"):
            folder = args.output / "samples" / role
            folder.mkdir(parents=True, exist_ok=True)
            data = datasets[role]
            for index in range(len(data)):
                line = data.ids[index]
                identity = image_identity(line)
                destination = folder / f"{identity}.npz"
                if destination.exists():
                    with np.load(destination, allow_pickle=False) as item:
                        if (item["checkpoint_sha256"].item() != checkpoint_hash or
                                item["prototype_sha256"].item() != prototype_hash or
                                item["protocol_sha256"].item() != signature["protocol_sha256"] or
                                item["image_id"].item() != identity or item["role"].item() != role):
                            raise ValueError("Existing pixel cache belongs to another run")
                else:
                    image, mask, _ = data[index]
                    mask_array = mask.numpy()
                    h, w = mask_array.shape
                    valid = np.flatnonzero(mask_array.ravel() != 255)
                    if not len(valid):
                        raise ValueError("No valid evaluation pixels in image")
                    count = min(protocol["sampling"]["sensitivity_per_image"], len(valid))
                    selected = fixed_rng("diagnostic_pixels", identity).choice(valid, count, replace=False)
                    yx = np.stack((selected//w, selected % w), 1)
                    logits = model(image[None].to(args.device), False)[0]
                    features = sample_decoder_features(captured["features"], torch.from_numpy(yx), (h, w))
                    sampled_logits = logits[:, yx[:, 0], yx[:, 1]].T
                    similarities = features@prototypes.T
                    scores = score_prediction(sampled_logits, similarities)
                    ground_truth = mask_array.ravel()[selected].astype(np.int64)
                    values = {"score__"+name:value.cpu().numpy() for name, value in scores.items() if value.ndim == 1}
                    temporary = destination.with_suffix(".partial.npz")
                    np.savez_compressed(temporary, image_id=np.asarray(identity), role=np.asarray(role),
                        original_size=np.asarray([h, w]), valid_pixel_count=np.asarray(len(valid)),
                        pixel_indices=selected, yx=yx, logits=sampled_logits.cpu().numpy(),
                        features=features.cpu().numpy(), prototype_similarities=similarities.cpu().numpy(),
                        GT=ground_truth, correct=(scores["pseudo_label"].cpu().numpy() == ground_truth),
                        boundary=boundary_pixels(mask_array).ravel()[selected],
                        checkpoint_sha256=np.asarray(checkpoint_hash), prototype_sha256=np.asarray(prototype_hash),
                        protocol_sha256=np.asarray(signature["protocol_sha256"]), **values)
                    temporary.replace(destination)
                all_records.append(dict(role=role, image_id=identity, file=str(destination.relative_to(args.output)),
                                        bytes=destination.stat().st_size, sha256=digest(destination)))
                if (index+1) % 25 == 0:
                    atomic_json(args.output / "export_status.json", dict(state="running", phase=role+"_cache", images=index+1,
                                target=len(data), signature=signature, updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    hook.remove()
    manifest = dict(signature=signature, prototype_sha256=prototype_hash, images={role:len(datasets[role]) for role in datasets},
                    files=all_records, torch=torch.__version__, cuda=torch.version.cuda,
                    device=torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else "cpu",
                    sampling=protocol["sampling"], boundary_definition="different valid GT classes within Manhattan distance 3, ignoring void",
                    role_isolation="only D_sup builds prototypes; only D_rel may fit verifiers; validation GT is evaluation-only")
    atomic_json(args.output / "manifest.json", manifest)
    complete = dict(state="complete", signature=signature, manifest_sha256=digest(args.output / "manifest.json"),
                    completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), images=manifest["images"])
    atomic_json(args.output / "COMPLETE.json", complete)
    atomic_json(args.output / "export_status.json", complete)
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
