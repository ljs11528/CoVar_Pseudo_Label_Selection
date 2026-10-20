"""Finite queue: verified data -> CPU/CUDA invariants -> real smoke -> teacher.

No new method training is launched. P1 frozen analyses are separate dependent
jobs; P2 remains gated by the preregistered evidence requirements.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .prepare_data import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--source-upload", required=True, type=Path)
    parser.add_argument("--pretrained", required=True, type=Path)
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--deadline-hours", type=float, default=96)
    args = parser.parse_args()
    root, release = args.root.resolve(), args.release.resolve()
    repo, partition = release / "repo", release / "prepared_partition"
    upstream = repo / "CSL+CoVar"
    state_file = root / "experiments/preparation_status.json"
    state = dict(state="running", scope="P0 and formal preparation teacher seed42 only; no G0-G7 end-to-end expansion",
                 pid=os.getpid(), source_manifest=json.loads((repo / "source_manifest.json").read_text()),
                 protocol_sha256=digest(repo / "research/label_anchored_competition/protocol.json"), completed=[],
                 started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), deadline_hours=args.deadline_hours)
    deadline = time.monotonic()+args.deadline_hours*3600
    def update(**values):
        state.update(values, updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        temporary = state_file.with_suffix(".partial.json")
        temporary.write_text(json.dumps(state, indent=2)+"\n")
        temporary.replace(state_file)
    def run(name, module, extra, completion):
        if completion.exists():
            if json.loads(completion.read_text()).get("state") != "complete":
                raise ValueError("Existing completion record is not complete")
            state["completed"].append(name)
            update(phase=name+"_already_complete")
            return
        command = [sys.executable, "-u", "-m", "research.label_anchored_competition."+module, *map(str, extra)]
        log_path = root / "logs" / (name+".log")
        with log_path.open("ab") as log:
            child = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            update(phase=name, command=command, child_pid=child.pid, log=str(log_path))
            while child.poll() is None:
                if time.monotonic() > deadline:
                    os.killpg(child.pid, signal.SIGTERM)
                    raise TimeoutError("Finite queue deadline reached")
                time.sleep(5)
        if child.returncode != 0 or not completion.exists():
            raise RuntimeError(f"{name} failed with code {child.returncode}; see {log_path}")
        if json.loads(completion.read_text()).get("state") != "complete":
            raise ValueError("Command exited without complete state")
        state["completed"].append(name)
        update(phase=name+"_complete")
    update(phase="data_wait_and_prepare")
    try:
        run("dataset", "bootstrap_dataset", ["--root", root, "--source-upload", args.source_upload,
            "--prepared-partition", partition], root / "data/COMPLETE.json")
        run("selfcheck_server_cpu", "run_selfcheck", ["--output", root / "logs/selfcheck_server_cpu.json"], root / "logs/selfcheck_server_cpu.json")
        run("cuda_preflight", "cuda_preflight", ["--output", root / "logs/cuda_preflight.json"], root / "logs/cuda_preflight.json")
        usage = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        values = [tuple(map(int, line.split(","))) for line in usage.splitlines()]
        if len(values) != args.devices or any(memory > 1024 or utilization > 5 for memory, utilization in values):
            raise RuntimeError("GPU allocation no longer matches the idle devices audited for this queue")
        common = ["--upstream", upstream, "--prepared-partition", partition,
                  "--data-root", root / "data/VOCdevkit/VOC2012", "--pretrained", args.pretrained,
                  "--devices", args.devices, "--workers", 4, "--seed", 42]
        smoke = root / "experiments/teacher_smoke_seed42"
        run("teacher_smoke", "train_teacher", [*common, "--smoke", "--output", smoke], smoke / "COMPLETE.json")
        run("teacher_smoke_audit", "audit_teacher_smoke", ["--run", smoke, "--upstream", upstream,
            "--data-root", root / "data/VOCdevkit/VOC2012", "--prepared-partition", partition,
            "--pretrained", args.pretrained, "--output", root / "logs/teacher_smoke_audit.json"], root / "logs/teacher_smoke_audit.json")
        formal = root / "experiments/voc_Dsup_mc_seed42"
        resume = formal / "checkpoints/last.ckpt"
        run("formal_teacher_seed42", "train_teacher", [*common, "--output", formal, *(["--resume", resume] if resume.exists() else [])], formal / "COMPLETE.json")
        update(state="complete", phase="preparation_teacher_ready")
    except Exception as exc:
        update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
