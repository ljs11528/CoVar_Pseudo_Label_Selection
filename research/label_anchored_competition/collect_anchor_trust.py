"""Hash-check all completed audit files; transport only small aggregate records."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


REMOTE = r'''
import sys,json,hashlib,io,tarfile,datetime
from pathlib import Path
runs=Path(sys.argv[1]).resolve()
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
complete=json.loads((runs/'COMPLETE.json').read_text())
if complete['state']!='complete':raise ValueError('Run incomplete')
files={};verified={}
for stage,expected_receipt in complete['stage_receipts'].items():
 folder=runs/('seed42_'+stage)
 if digest(folder/'COMPLETE.json')!=expected_receipt:raise ValueError('Stage receipt changed')
 receipt=json.loads((folder/'COMPLETE.json').read_text())
 for name,meta in receipt['files'].items():
  path=folder/name
  if path.is_symlink() or not path.resolve().is_relative_to(folder):raise ValueError('Unsafe input path')
  if path.stat().st_size!=meta['bytes'] or digest(path)!=meta['sha256']:raise ValueError('Completed artifact changed: '+name)
  verified[str(path.relative_to(runs))]=meta
  if path.suffix in ('.csv','.json'):files[str(path.relative_to(runs))]=path
 files[str((folder/'COMPLETE.json').relative_to(runs))]=folder/'COMPLETE.json'
for name in ('COMPLETE.json','gate.json','ALL_FITTING_COMPLETE.json'):
 files[name]=runs/name
tag=runs.name.removeprefix('runs_')
for name in ('launch_'+tag+'.json','device_smoke_'+tag+'.json'):
 files[name]=runs.parent/name
snapshot=dict(captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(runs),files={},
 verified_remote_files=verified,scope='all files verified; only aggregate CSV/JSON transported; pixel arrays remain remote')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name,path in sorted(files.items()):
  data=path.read_bytes()
  if len(data)>25*1024*1024:raise ValueError('Oversized aggregate artifact')
  snapshot['files'][name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
  entry=tarfile.TarInfo(name);entry.size=len(data);archive.addfile(entry,io.BytesIO(data))
 data=(json.dumps(snapshot,indent=2)+'\n').encode()
 entry=tarfile.TarInfo('snapshot_receipt.json');entry.size=len(data);archive.addfile(entry,io.BytesIO(data))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('host','key','known-hosts','proxy-command','runs'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    command=['ssh','-i',args.key,'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=20',
             '-o','UserKnownHostsFile='+args.known_hosts,'-o','StrictHostKeyChecking=yes',
             '-o','ProxyCommand='+args.proxy_command,args.host,
             '/opt/conda/bin/python -c '+shlex.quote(REMOTE)+' '+shlex.quote(args.runs)]
    payload=subprocess.check_output(command)
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
        receipt=json.loads(archive.extractfile('snapshot_receipt.json').read())
        if set(m.name for m in archive.getmembers())!=set(receipt['files'])|{'snapshot_receipt.json'}:
            raise ValueError('Snapshot membership mismatch')
        args.output.mkdir(parents=True)
        for member in archive.getmembers():
            name=Path(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts:
                raise ValueError('Unsafe snapshot member')
            data=archive.extractfile(member).read()
            if member.name in receipt['files']:
                meta=receipt['files'][member.name]
                if len(data)!=meta['bytes'] or hashlib.sha256(data).hexdigest()!=meta['sha256']:
                    raise ValueError('Transport checksum mismatch')
            path=args.output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    print(json.dumps(dict(output=str(args.output),files=len(receipt['files']),
                         verified_remote_files=len(receipt['verified_remote_files']),captured_at_utc=receipt['captured_at_utc'])))


if __name__=='__main__':
    main()
