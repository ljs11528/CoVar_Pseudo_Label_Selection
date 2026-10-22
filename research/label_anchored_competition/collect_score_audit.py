"""Verify completed remote audit artifacts and copy only small records."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


REMOTE=r'''
import sys,json,hashlib,io,tarfile,datetime
from pathlib import Path
root=Path(sys.argv[1]).resolve()
runs=root/sys.argv[2]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
files={}; verified={}
for stage in ['early','middle','late']:
 folder=runs/('seed42_'+stage)
 complete=json.loads((folder/'COMPLETE.json').read_text())
 if complete['state']!='complete':raise ValueError('Unfinished stage')
 for name,meta in complete['files'].items():
  path=folder/name
  if path.is_symlink() or not path.resolve().is_relative_to(folder):raise ValueError('Unsafe path')
  if path.stat().st_size!=meta['bytes'] or digest(path)!=meta['sha256']:raise ValueError('Completion hash mismatch')
  verified[str(path.relative_to(runs))]=meta
  if path.suffix in ['.json','.csv']:files[str(path.relative_to(runs))]=path
 files[str((folder/'COMPLETE.json').relative_to(runs))]=folder/'COMPLETE.json'
tag=runs.name.removeprefix('runs_')
for name in ['launch_'+tag+'.json','device_smoke_'+tag+'.json']:
 files[name]=root/name
for path in root.glob('runs_*/seed42_*/ABORTED_IMPLEMENTATION.json'):
 files['aborted_attempts/'+path.parent.parent.name+'/'+path.parent.name+'.json']=path
receipt=dict(captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(runs),files={},
 scope='All completed artifacts hash-verified on server; only CSV/JSON transported; tensors remain remote',verified_remote_files=verified)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name,path in sorted(files.items()):
  data=path.read_bytes()
  if len(data)>25*1024*1024:raise ValueError('Unexpected small artifact size')
  receipt['files'][name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
  entry=tarfile.TarInfo(name);entry.size=len(data);archive.addfile(entry,io.BytesIO(data))
 data=(json.dumps(receipt,indent=2)+'\n').encode()
 entry=tarfile.TarInfo('snapshot_receipt.json');entry.size=len(data);archive.addfile(entry,io.BytesIO(data))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('host','key','known-hosts','proxy-command','root','runs'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if '/' in args.runs or args.runs in ('.','..'):
        raise ValueError('Expected a run directory name')
    command=['ssh','-i',args.key,'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=20',
        '-o','UserKnownHostsFile='+args.known_hosts,'-o','ProxyCommand='+args.proxy_command,args.host,
        '/opt/conda/bin/python -c '+shlex.quote(REMOTE)+' '+shlex.quote(args.root)+' '+shlex.quote(args.runs)]
    payload=subprocess.check_output(command)
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
        members=archive.getmembers()
        receipt=json.loads(archive.extractfile('snapshot_receipt.json').read())
        if set(m.name for m in members)!=set(receipt['files'])|{'snapshot_receipt.json'}:
            raise ValueError('Snapshot membership mismatch')
        args.output.mkdir(parents=True)
        for member in members:
            name=Path(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts:
                raise ValueError('Unsafe archive member')
            data=archive.extractfile(member).read()
            if member.name in receipt['files']:
                expected=receipt['files'][member.name]
                if len(data)!=expected['bytes'] or hashlib.sha256(data).hexdigest()!=expected['sha256']:
                    raise ValueError('Transport checksum mismatch')
            path=args.output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    print(json.dumps(dict(output=str(args.output),files=len(receipt['files']),
        verified_remote_files=len(receipt['verified_remote_files']),captured_at_utc=receipt['captured_at_utc'])))


if __name__=='__main__':
    main()
