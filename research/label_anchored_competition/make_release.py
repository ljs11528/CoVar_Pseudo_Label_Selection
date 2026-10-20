"""Build a small checksummed release from committed source and prepared splits."""
import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-partition", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "-z", commit, "--", "research", "CSL+CoVar"], cwd=root).decode().split("\0")
    content = {}
    for name in filter(None, tracked):
        path = root / name
        if path.suffix in (".py", ".yaml", ".yml", ".md", ".json", ".csv") or path.name in (".gitignore", "LICENSE"):
            # Read the commit itself, never a possibly modified working copy.
            content["repo/"+name] = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
    source_identity = dict(commit=commit, created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                           files={name.removeprefix("repo/"):hashlib.sha256(data).hexdigest()
                                  for name, data in sorted(content.items())})
    content["repo/source_manifest.json"] = (json.dumps(source_identity, indent=2)+"\n").encode()
    for path in sorted(args.prepared_partition.rglob("*")):
        if path.is_file():
            content["prepared_partition/"+str(path.relative_to(args.prepared_partition))] = path.read_bytes()
    manifest = dict(commit=commit, files={name:dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                                          for name, data in sorted(content.items())})
    content["MANIFEST.json"] = (json.dumps(manifest, indent=2)+"\n").encode()
    if args.output.exists():
        raise FileExistsError(args.output)
    with tarfile.open(args.output, "w:gz") as archive:
        for name, data in sorted(content.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    print(json.dumps(dict(commit=commit, files=len(content), bytes=args.output.stat().st_size,
                          sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(), output=str(args.output))))


if __name__ == "__main__":
    main()
