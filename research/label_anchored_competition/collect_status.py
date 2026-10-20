"""Collect checksummed small status/results; leave images and tensors remote."""
import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


REMOTE = r'''
from pathlib import Path
import sys,io,tarfile,json,hashlib,datetime,subprocess
root=Path(sys.argv[1]).resolve()
files={}
for folder in ['logs','downloads','data','experiments']:
 for path in (root/folder).glob('*.json'):
  if path.is_file():files[str(path.relative_to(root))]=path
for run in (root/'experiments').glob('*'):
 if not run.is_dir():continue
 for path in run.glob('*.json'):
  if path.is_file():files[str(path.relative_to(root))]=path
 for name in ['analysis_1024','analysis_4096']:
  for path in (run/name).glob('*'):
   if path.is_file() and path.suffix in ['.csv','.json'] and path.name!='risk_coverage.csv':files[str(path.relative_to(root))]=path
receipt={'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),
 'scope':'small status and results snapshot; only verified COMPLETE markers establish completion','files':{},
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True)}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name,path in sorted(files.items()):
  if path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('Unsafe status path')
  data=path.read_bytes()
  if len(data)>25*1024*1024:raise ValueError('Unexpectedly large status file')
  receipt['files'][name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
  item=tarfile.TarInfo(name);item.size=len(data);archive.addfile(item,io.BytesIO(data))
 data=(json.dumps(receipt,indent=2)+'\n').encode()
 item=tarfile.TarInfo('snapshot_receipt.json');item.size=len(data);archive.addfile(item,io.BytesIO(data))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--known-hosts", required=True)
    parser.add_argument("--proxy-command")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    command = ["ssh", "-i", args.key, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
               "-o", "UserKnownHostsFile="+args.known_hosts]
    if args.proxy_command:
        command.extend(["-o", "ProxyCommand="+args.proxy_command])
    command.extend([args.host, "/opt/conda/bin/python -c "+shlex.quote(REMOTE)+" "+shlex.quote(args.root)])
    result = subprocess.run(command, capture_output=True, check=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:gz") as archive:
        members = archive.getmembers()
        receipt = json.loads(archive.extractfile("snapshot_receipt.json").read())
        if {x.name for x in members} != set(receipt["files"]) | {"snapshot_receipt.json"}:
            raise ValueError("Snapshot membership differs")
        for member in members:
            name = Path(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise ValueError("Unsafe snapshot member")
            payload = archive.extractfile(member).read()
            if member.name in receipt["files"]:
                expected = receipt["files"][member.name]
                if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                    raise ValueError("Snapshot checksum differs")
            path = args.output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    print(json.dumps(dict(output=str(args.output.resolve()), payload_files=len(receipt["files"]), captured_at_utc=receipt["captured_at_utc"])))


if __name__ == "__main__":
    main()
