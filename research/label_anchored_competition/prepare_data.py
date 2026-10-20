"""Validate the supplied image split and create the budget-internal verifier split.

Only labeled training masks are read. No validation or unlabeled masks enter
the support audit or any split decision. Split choice is fixed before the audit.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

import numpy as np
from PIL import Image


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_revision(directory):
    """Resolve either a checked-out Git tree or an immutable deployed release."""
    directory = Path(directory).resolve()
    for candidate in (directory, *directory.parents):
        manifest = candidate / "source_manifest.json"
        if manifest.exists():
            return json.loads(manifest.read_text())["commit"]
        if (candidate / ".git").exists():
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=candidate, text=True).strip()
    raise ValueError("No Git checkout or verified source release manifest")


def image_identity(line):
    parts = line.split()
    if len(parts) != 2:
        raise ValueError("Expected one image and one mask per split row")
    for name in parts:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe dataset-relative path")
    image_id = PurePosixPath(parts[0]).stem
    if image_id != PurePosixPath(parts[1]).stem:
        raise ValueError("Image/mask identities differ")
    return image_id


def checked_rows(path, expected_hash):
    if digest(path) != expected_hash:
        raise ValueError(f"Input split hash differs: {path.name}")
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    mapping = {image_identity(line): line for line in lines}
    if len(mapping) != len(lines):
        raise ValueError("Duplicate image identity in source split")
    return mapping


def partition(labeled, fraction, seed):
    ordered = sorted(labeled, key=lambda image_id: (
        hashlib.sha256(f"{seed}|{image_id}".encode()).digest(), image_id))
    count = round(len(ordered) * fraction)
    if not 0 < count < len(ordered):
        raise ValueError("Both partitions must contain images")
    return sorted(ordered[count:]), sorted(ordered[:count])


def disjoint_partitions(groups):
    names = list(groups)
    overlaps = {}
    for i, name in enumerate(names):
        for other in names[i+1:]:
            overlap = set(groups[name]) & set(groups[other])
            overlaps[f"{name}__{other}"] = len(overlap)
    if any(overlaps.values()):
        raise ValueError(f"Image leakage: {overlaps}")
    return overlaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--mask-archive", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    cfg = protocol["dataset"]
    hashes = cfg["split_sha256"]
    labeled = checked_rows(args.split_root / "1_8/labeled.txt", hashes["labeled"])
    unlabeled = checked_rows(args.split_root / "1_8/unlabeled.txt", hashes["unlabeled"])
    validation = checked_rows(args.split_root / "val.txt", hashes["validation"])
    if (len(labeled), len(unlabeled), len(validation)) != (
            cfg["label_budget"], cfg["unlabeled_images"], cfg["diagnostic_images"]):
        raise ValueError("Counts differ from locked protocol")
    split_cfg = protocol["partition"]
    supervised, reliability = partition(labeled, split_cfg["verifier_fraction"], split_cfg["seed"])
    if (len(supervised), len(reliability)) != (
            split_cfg["supervised_images"], split_cfg["verifier_images"]):
        raise ValueError("Budget-internal split sizes differ from protocol")
    overlaps = disjoint_partitions(dict(D_sup=supervised, D_rel=reliability,
                                        D_U=list(unlabeled), validation=list(validation)))
    support = []
    per_image = []
    with zipfile.ZipFile(args.mask_archive) as archive:
        names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member")
        for group, image_ids in (("D_sup", supervised), ("D_rel", reliability)):
            pixel_counts = np.zeros(cfg["classes"], dtype=np.int64)
            image_counts = np.zeros_like(pixel_counts)
            for image_id in image_ids:
                mask_path = labeled[image_id].split()[1]
                with Image.open(io.BytesIO(archive.read(mask_path))) as image:
                    mask = np.asarray(image)
                if mask.ndim != 2:
                    raise ValueError(f"Unexpected mask shape: {image_id}")
                values, counts = np.unique(mask, return_counts=True)
                if any(int(v) not in range(cfg["classes"]) and int(v) != cfg["ignore_index"] for v in values):
                    raise ValueError(f"Invalid class ID: {image_id}")
                present = []
                for value, count in zip(values, counts):
                    if value == cfg["ignore_index"]:
                        continue
                    pixel_counts[int(value)] += int(count)
                    image_counts[int(value)] += 1
                    present.append(int(value))
                per_image.append(dict(image_id=image_id, partition=group, classes=present))
            support.extend(dict(partition=group, class_id=c, images=int(image_counts[c]),
                                pixels=int(pixel_counts[c])) for c in range(cfg["classes"]))
    absent = [x["class_id"] for x in support if x["partition"] == "D_sup" and x["images"] == 0]
    if absent:
        raise ValueError(f"No supervised support for prototype classes: {absent}")
    # Materialize only after all safety and fixed-split checks have passed.
    args.output.mkdir(parents=True, exist_ok=False)
    splits = args.output / "splits"
    (splits / "1_8").mkdir(parents=True)
    for path, ids, source in [(splits / "1_8/labeled.txt", supervised, labeled),
                              (splits / "rel.txt", reliability, labeled),
                              (splits / "1_8/unlabeled.txt", list(unlabeled), unlabeled),
                              (splits / "val.txt", list(validation), validation)]:
        path.write_text("".join(source[image_id]+"\n" for image_id in ids))
    # Preserve original stream order and exact public validation identity files.
    (splits / "1_8/unlabeled.txt").write_bytes((args.split_root / "1_8/unlabeled.txt").read_bytes())
    (splits / "val.txt").write_bytes((args.split_root / "val.txt").read_bytes())
    (args.output / "partitions.json").write_text(json.dumps(dict(
        D_sup=supervised, D_rel=reliability, image_labels_used="training D_L only",
        per_image_class_support=per_image), indent=2)+"\n")
    with (args.output / "class_support.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["partition", "class_id", "images", "pixels"])
        writer.writeheader()
        writer.writerows(support)
    commit = source_revision(Path(__file__).parent)
    audit = dict(state="complete", scope="data isolation and labeled-mask support; no model or method performance",
                 completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 source_commit=commit, tool_sha256=digest(__file__), protocol_sha256=digest(args.protocol),
                 original_split_sha256=hashes,
                 counts=dict(D_sup=len(supervised), D_rel=len(reliability), D_U=len(unlabeled), validation=len(validation)),
                 overlaps=overlaps, mask_archive_sha256=digest(args.mask_archive),
                 classes_with_supervised_support=cfg["classes"]-len(absent), class_support=support,
                 labeled_image_support_min=min(x["images"] for x in support if x["partition"] == "D_sup"),
                 outputs={str(path.relative_to(args.output)): dict(sha256=digest(path), bytes=path.stat().st_size)
                          for path in sorted(args.output.rglob("*")) if path.is_file()},
                 limitations=["Training images and backbone features not read or verified here.",
                              "Validation mask provenance must be audited before formal export.",
                              "Class support is a labeled data inventory, not prototype quality."])
    (args.output / "COMPLETE.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({k:audit[k] for k in ("state", "counts", "classes_with_supervised_support", "labeled_image_support_min")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
