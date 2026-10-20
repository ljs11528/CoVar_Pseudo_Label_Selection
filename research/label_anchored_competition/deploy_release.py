"""Deploy a verified, immutable source release; never overwrite previous runs."""
import argparse
from pathlib import Path
import shlex
import subprocess


RECEIVER = r'''
import sys,io,tarfile,json,hashlib,re
from pathlib import Path
root=Path(sys.argv[1]).resolve()
data=sys.stdin.buffer.read(100*1024*1024+1)
if len(data)>100*1024*1024:raise ValueError('Unexpectedly large release')
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 members=archive.getmembers()
 manifest=json.loads(archive.extractfile('MANIFEST.json').read())
 commit=manifest['commit']
 if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Invalid commit identity')
 if set(x.name for x in members)!=set(manifest['files'])|{'MANIFEST.json'}:raise ValueError('Manifest membership differs')
 target=root/'releases'/commit
 staging=root/'releases'/('.'+commit+'.partial')
 if target.exists():
  record=json.loads((target/'RELEASE_RECEIPT.json').read_text())
  if record['archive_sha256']!=hashlib.sha256(data).hexdigest():raise ValueError('Existing release differs')
  print(json.dumps({'existing_identical_release':str(target)}));sys.exit(0)
 staging.mkdir(parents=True,exist_ok=False)
 for member in members:
  name=Path(member.name)
  if not member.isfile() or name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe release member')
  payload=archive.extractfile(member).read()
  if member.name in manifest['files']:
   expected=manifest['files'][member.name]
   if len(payload)!=expected['bytes'] or hashlib.sha256(payload).hexdigest()!=expected['sha256']:raise ValueError('Release hash differs')
  path=staging/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
 record={'commit':commit,'files':len(members),'archive_sha256':hashlib.sha256(data).hexdigest()}
 (staging/'RELEASE_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n')
 staging.replace(target)
 print(json.dumps(dict(record,release=str(target))))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--proxy-command")
    parser.add_argument("--known-hosts", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    command = ["ssh", "-i", str(args.key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=20", "-o", "UserKnownHostsFile="+args.known_hosts]
    if args.proxy_command:
        command.extend(["-o", "ProxyCommand="+args.proxy_command])
    command.extend([args.host, "/opt/conda/bin/python -c "+shlex.quote(RECEIVER)+" "+shlex.quote(args.root)])
    subprocess.run(command, input=args.archive.read_bytes(), check=True)


if __name__ == "__main__":
    main()
