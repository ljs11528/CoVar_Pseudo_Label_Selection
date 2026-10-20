"""Prepare a separate verified VOC dataset after the official download finishes."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
import zipfile

from PIL import Image

from .prepare_data import digest, disjoint_partitions, image_identity


VOC_BYTES = 1999639040
VOC_MD5 = "6cd6e144f989b92b3379bac3b3de84fd"
MASK_SHA256 = "06a7befb7565fe998c18d4d6b2d5e9cb4967f1bea4cb89321eee107eef1c99d1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-upload", type=Path, required=True)
    parser.add_argument("--prepared-partition", type=Path, required=True)
    parser.add_argument("--wait-hours", type=float, default=6)
    args = parser.parse_args()
    target = args.root / "data/VOCdevkit/VOC2012"
    complete_path = args.root / "data/COMPLETE.json"
    if complete_path.exists():
        raise FileExistsError("Dataset already prepared; preserve the receipt")
    archive = args.root / "downloads/VOCtrainval_11-May-2012.tar"
    partial = archive.with_suffix(".tar.part")
    deadline = time.monotonic()+3600*args.wait_hours
    while not archive.exists():
        if partial.exists() and partial.stat().st_size == VOC_BYTES:
            hasher = hashlib.md5()
            with partial.open("rb") as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b""):
                    hasher.update(block)
            if hasher.hexdigest() != VOC_MD5:
                raise ValueError("Official VOC archive MD5 differs")
            partial.replace(archive)
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Official VOC download not ready")
        time.sleep(5)
    hasher = hashlib.md5()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            hasher.update(block)
    if archive.stat().st_size != VOC_BYTES or hasher.hexdigest() != VOC_MD5:
        raise ValueError("Official archive checksum/size mismatch")
    args.root.joinpath("data").mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for item in members:
            path = PurePosixPath(item.name)
            if path.is_absolute() or ".." in path.parts or not (item.isfile() or item.isdir()):
                raise ValueError("Unsafe official archive member")
            if not path.parts or path.parts[0] != "VOCdevkit":
                raise ValueError("Unexpected official archive root")
        tar.extractall(args.root / "data", members=members)
    original = target / "SegmentationClassOfficial"
    shutil.copytree(target / "SegmentationClass", original, dirs_exist_ok=True)
    mask_zip = args.root / "assets/SegmentationClass.zip"
    if not mask_zip.exists():
        with zipfile.ZipFile(args.source_upload) as uploaded:
            name = "2026IJCV/code/SegmentationClass.zip"
            with uploaded.open(name) as source, mask_zip.open("wb") as output:
                shutil.copyfileobj(source, output)
    if digest(mask_zip) != MASK_SHA256:
        raise ValueError("Uploaded augmented masks differ from audited input")
    with zipfile.ZipFile(mask_zip) as masks:
        for item in masks.infolist():
            if item.is_dir():
                continue
            path = PurePosixPath(item.filename)
            if len(path.parts) != 2 or path.parts[0] != "SegmentationClass" or path.suffix != ".png":
                raise ValueError("Unexpected augmented mask member")
            with masks.open(item) as source, (target / "SegmentationClass" / path.name).open("wb") as output:
                shutil.copyfileobj(source, output)
    splits = args.prepared_partition / "splits"
    groups = {}
    validation_mask_hashes = {}
    for name, relative in [("D_sup", "1_8/labeled.txt"), ("D_rel", "rel.txt"),
                            ("D_U", "1_8/unlabeled.txt"), ("validation", "val.txt")]:
        lines = (splits / relative).read_text().splitlines()
        groups[name] = [image_identity(line) for line in lines]
        for line in lines:
            image_path, mask_path = line.split()
            if name == "validation":
                official = original / Path(mask_path).name
                shutil.copy2(official, target / mask_path)
                validation_mask_hashes[Path(mask_path).name] = digest(official)
            with Image.open(target / image_path) as image, Image.open(target / mask_path) as mask:
                if image.size != mask.size:
                    raise ValueError("Image and mask dimensions differ")
        print(json.dumps(dict(partition=name, images=len(lines), checked=True)), flush=True)
    overlaps = disjoint_partitions(groups)
    official_val = set((target / "ImageSets/Segmentation/val.txt").read_text().split())
    if set(groups["validation"]) != official_val:
        raise ValueError("Diagnostic validation identities differ from official VOC")
    receipt = dict(state="complete", completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   archive_bytes=VOC_BYTES, archive_md5=VOC_MD5, augmented_mask_sha256=MASK_SHA256,
                   role_counts={name:len(ids) for name, ids in groups.items()}, overlaps=overlaps,
                   jpeg_images=len(list((target / "JPEGImages").glob("*.jpg"))),
                   validation_masks="restored from verified official archive before diagnostics",
                   validation_mask_identity_sha256=hashlib.sha256(json.dumps(validation_mask_hashes, sort_keys=True).encode()).hexdigest(),
                   partition_receipt_sha256=digest(args.prepared_partition / "COMPLETE.json"),
                   tool_sha256=digest(__file__), scope="data preparation only, no method performance")
    complete_path.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
