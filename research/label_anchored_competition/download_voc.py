"""Finite, resumable official VOC archive download with fixed MD5 validation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

from .bootstrap_dataset import VOC_BYTES, VOC_MD5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--url", default="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar")
    args = parser.parse_args()
    if not 1 <= args.connections <= 8:
        parser.error("Use at most eight connections")
    directory = args.root / "downloads"
    directory.mkdir(parents=True, exist_ok=True)
    parts = directory / "VOC_chunks"
    parts.mkdir(exist_ok=True)
    target = directory / "VOCtrainval_11-May-2012.tar"
    if target.exists():
        raise FileExistsError("Preserve the verified archive")
    chunk_size = 32*1024*1024
    count = (VOC_BYTES+chunk_size-1)//chunk_size
    started = time.monotonic()

    def fetch(index):
        start = index*chunk_size
        end = min(VOC_BYTES, start+chunk_size)-1
        path = parts / f"chunk_{index:03d}"
        length = end-start+1
        if path.exists() and path.stat().st_size == length:
            return length
        partial = path.with_suffix(".part")
        for retry in range(4):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > length:
                raise ValueError("Unexpected oversized partial range")
            if offset == length:
                partial.replace(path)
                return length
            request = urllib.request.Request(args.url, headers={"Range":f"bytes={start+offset}-{end}"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    expected = f"bytes {start+offset}-{end}/{VOC_BYTES}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server range identity differs")
                    with partial.open("ab") as output:
                        shutil.copyfileobj(response, output, length=1024*1024)
                if partial.stat().st_size != length:
                    raise ValueError("Range response ended at unexpected size")
                partial.replace(path)
                return length
            except Exception:
                if retry == 3:
                    raise
                time.sleep(2**retry)

    with ThreadPoolExecutor(max_workers=args.connections) as pool:
        finished = 0
        for future in as_completed([pool.submit(fetch, index) for index in range(count)]):
            finished += future.result()
            progress = dict(state="running", completed_chunks_bytes=finished, total_bytes=VOC_BYTES,
                            elapsed_seconds=time.monotonic()-started,
                            updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            temporary = args.root / "logs/download_progress.partial.json"
            temporary.write_text(json.dumps(progress, indent=2)+"\n")
            temporary.replace(args.root / "logs/download_progress.json")
            print(json.dumps(progress), flush=True)
    # The dataset preparer watches .tar/.tar.part; publish only after checksum.
    assembled = target.with_suffix(".tar.assembling")
    if assembled.exists():
        raise FileExistsError("Do not overwrite a previous assembled archive")
    hasher = hashlib.md5()
    with assembled.open("wb") as output:
        for index in range(count):
            with (parts / f"chunk_{index:03d}").open("rb") as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b""):
                    hasher.update(block)
                    output.write(block)
    if assembled.stat().st_size != VOC_BYTES or hasher.hexdigest() != VOC_MD5:
        raise ValueError("Reassembled archive MD5/size mismatch")
    assembled.replace(target)
    receipt = dict(state="complete", bytes=VOC_BYTES, md5=VOC_MD5, url=args.url,
                   elapsed_seconds=time.monotonic()-started, completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    (args.root / "downloads/COMPLETE.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
