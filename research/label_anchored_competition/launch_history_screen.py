"""Launch one finite CPU-only history screen from a verified immutable release."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from .prepare_data import digest, source_revision
from .run_score_audit import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    repo = package.parents[1]
    commit = source_revision(package)
    source = json.loads((repo/'source_manifest.json').read_text())
    if source['commit'] != commit:
        raise ValueError('Immutable source identity mismatch')
    for name, expected in source['files'].items():
        if digest(repo/name) != expected:
            raise ValueError('Released source changed: '+name)
    audit = args.root/'audits/historical_competition_20261023'; audit.mkdir(parents=True, exist_ok=True)
    tag = commit[:8]
    output = audit/('runs_'+tag)
    launch = audit/('launch_'+tag+'.json')
    smoke_path = audit/('server_smoke_'+tag+'.json')
    smoke = json.loads(smoke_path.read_text())
    if smoke['state'] != 'passed' or smoke['source_commit'] != commit:
        raise ValueError('Matching server smoke must pass before launch')
    if output.exists() or launch.exists():
        raise FileExistsError('Never duplicate or overwrite a previous audit')
    command = [sys.executable, '-B', '-m', 'research.label_anchored_competition.run_history_screen',
               '--root', str(args.root.resolve()), '--output', str(output.resolve())]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    log_path = audit/('screen_'+tag+'.log')
    with log_path.open('xb') as log:
        process = subprocess.Popen(command, cwd=repo, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(source_commit=commit, pid=process.pid, hostname=socket.gethostname(),
        launched_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), command=command, cwd=str(repo),
        output=str(output), log=str(log_path), screen_hours_limit=2, no_teacher_updates=True,
        device='CPU', thread_limit=4, smoke_sha256=digest(smoke_path))
    write_json(launch, receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
