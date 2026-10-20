"""Finite P1 queue, with optional seed43 confirmation only after the fixed gate.

All exports/analyses wait for the seed42 teacher to finish. This queue never
launches the eight-group end-to-end matrix or silently changes a failed score.
"""
import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .export_frozen import atomic_json
from .prepare_data import digest


def csv_rows(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def decide_gate(outputs):
    primary = {stage:csv_rows(folder / "analysis_1024/primary_contrasts.csv") for stage, folder in outputs.items()}
    if any(len(rows) != 4 for rows in primary.values()):
        return dict(state="inconclusive", pass_to_confirmation=False, reason="Primary conflict support/intervals missing; no automatic expansion")
    late = primary["late"]
    positive_late = all(float(row["ci_low"]) > 0 and int(row["valid_replicates"]) >= 1900 for row in late)
    consistency = {row["contrast"]:sum(float(next(item for item in rows if item["contrast"] == row["contrast"])["delta_error_ap"]) > 0
                                      for rows in primary.values()) for row in late}
    negative_all = []
    uniform = None
    sensitivity = {}
    for stage, folder in outputs.items():
        intervals = csv_rows(folder / "analysis_1024/paired_image_bootstrap.csv")
        for row in intervals:
            if row["scope"] == "all" and row["score"] == "F_plus" and row["delta_vs"] == "F_base" and row["metric"] == "error_ap" and float(row["delta_ci_high"]) < 0:
                negative_all.append(stage)
            if stage == "late" and row["scope"] == "conflict" and row["score"] == "h_e" and row["delta_vs"] == "h_uniform" and row["metric"] == "error_ap":
                uniform = dict(ci_low=float(row["delta_ci_low"]), ci_high=float(row["delta_ci_high"]))
        points = csv_rows(folder / "analysis_4096/ranking.csv")
        local = {row["method"]:float(row["error_ap"]) for row in points if row["scope"] == "conflict"}
        sensitivity[stage] = local.get("F_plus", float("nan"))-local.get("F_base", float("nan"))
    uniform_supported = uniform is not None and uniform["ci_low"] > 0
    passed = bool(positive_late and all(value >= 2 for value in consistency.values()) and not negative_all and uniform_supported)
    return dict(state="candidate_requires_seed43_confirmation" if passed else "no_go_or_inconclusive", pass_to_confirmation=passed,
                primary_late_all_corrected_lower_bounds_positive=positive_late,
                positive_point_stages_by_contrast=consistency, clear_all_pixel_AP_degradation_stages=negative_all,
                late_uniform_control_95_CI=uniform, uniform_control_supported=uniform_supported,
                sensitivity_4096_conflict_increment={key:value if math.isfinite(value) else None for key, value in sensitivity.items()},
                sensitivity_has_no_separate_CI=True,
                limitations="One fixed training seed; image CIs condition on fitted models; this is a project gate, not a reliability theorem or seed-population guarantee",
                next_action="Run fixed seed43 preparation and diagnostics" if passed else "Preserve all results; stop expansion and interpret the failed/uncertain controls before changing the design")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-devices", type=int, default=4)
    parser.add_argument("--confirm-if-pass", action="store_true")
    parser.add_argument("--deadline-hours", type=float, default=96)
    args = parser.parse_args()
    root, release = args.root.resolve(), args.release.resolve()
    repo, partition = release / "repo", release / "prepared_partition"
    protocol = json.loads((repo / "research/label_anchored_competition/protocol.json").read_text())
    source = json.loads((repo / "source_manifest.json").read_text())
    deadline = time.monotonic()+args.deadline_hours*3600
    status_file = root / "experiments/frozen_queue_status.json"
    state = dict(state="running", pid=os.getpid(), source_commit=source["commit"],
                 protocol_sha256=digest(repo / "research/label_anchored_competition/protocol.json"),
                 completed=[], started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 scope="P1 only: same-prediction A/B/C/D, then seed43 only if gate passes; no G0-G7 runs")
    def update(**values):
        state.update(values, updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(status_file, state)
    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError("P1 finite queue deadline reached")
    def execute(name, module, options, complete):
        check_deadline()
        if complete.exists():
            stored = json.loads(complete.read_text())
            if stored.get("state") != "complete" or stored.get("signature", {}).get("synthetic_smoke"):
                raise ValueError("Existing result is incomplete or synthetic")
            state["completed"].append(name)
            update(phase=name+"_already_complete")
            return
        command = [sys.executable, "-u", "-m", "research.label_anchored_competition."+module, *map(str, options)]
        log_path = root / "logs" / (name+".log")
        with log_path.open("ab") as log:
            process = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            update(phase=name, child_pid=process.pid, command=command, log=str(log_path))
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    os.killpg(process.pid, signal.SIGTERM)
                    raise TimeoutError("P1 phase deadline reached")
                time.sleep(5)
        if process.returncode or not complete.exists():
            raise RuntimeError(f"{name} exited with {process.returncode}; see {log_path}")
        if json.loads(complete.read_text()).get("state") != "complete":
            raise RuntimeError("Phase exited without a complete result")
        state["completed"].append(name)
        update(phase=name+"_complete")
    def wait_teacher(seed):
        folder = root / "experiments" / f"voc_Dsup_mc_seed{seed}"
        marker = folder / "COMPLETE.json"
        update(phase=f"waiting_for_formal_teacher_seed{seed}", child_pid=None)
        while not marker.exists():
            check_deadline()
            if seed == 42:
                status = root / "experiments/preparation_status.json"
                if status.exists() and json.loads(status.read_text()).get("state") == "failed":
                    raise RuntimeError("Preparation queue failed; do not wait silently")
            time.sleep(30)
        record = json.loads(marker.read_text())
        if record.get("state") != "complete" or record.get("smoke") or record["global_step"] != protocol["teacher"]["total_optimizer_steps"] or record["signature"]["seed"] != seed:
            raise ValueError("Teacher completion is not the fixed formal trajectory")
        return folder
    def analyze_seed(seed):
        teacher = wait_teacher(seed)
        outputs = {}
        for stage, step in protocol["teacher"]["stage_steps"].items():
            folder = root / "experiments" / f"frozen_seed{seed}_{stage}_{source['commit'][:8]}"
            outputs[stage] = folder
            execute(f"seed{seed}_{stage}_export", "export_frozen", ["--upstream", repo / "CSL+CoVar",
                "--checkpoint", teacher / "checkpoints" / f"step_{step:06d}.ckpt", "--prepared-partition", partition,
                "--data-root", root / "data/VOCdevkit/VOC2012", "--output", folder, "--stage", stage,
                "--device", args.device], folder / "COMPLETE.json")
            for cap in (1024, 4096):
                execute(f"seed{seed}_{stage}_analysis_{cap}", "analyze_frozen", ["--export", folder,
                        "--cap", cap, "--device", args.device], folder / f"analysis_{cap}/COMPLETE.json")
        gate = decide_gate(outputs)
        gate.update(seed=seed, outputs={name:str(path) for name, path in outputs.items()}, source_commit=source["commit"],
                    completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(root / "experiments" / f"P1_gate_seed{seed}.json", gate)
        return gate
    try:
        update(phase="waiting_for_formal_teacher_seed42")
        first = analyze_seed(42)
        if first["pass_to_confirmation"] and args.confirm_if_pass:
            usage = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
            allocation = [tuple(map(int, row.split(","))) for row in usage.splitlines()]
            if len(allocation) != args.teacher_devices or any(memory > 1024 or utilization > 5 for memory, utilization in allocation):
                raise RuntimeError("Confirmation allocation no longer idle; do not occupy another job")
            teacher = root / "experiments/voc_Dsup_mc_seed43"
            resume = teacher / "checkpoints/last.ckpt"
            execute("confirmation_teacher_seed43", "train_teacher", ["--upstream", repo / "CSL+CoVar", "--prepared-partition", partition,
                "--data-root", root / "data/VOCdevkit/VOC2012", "--pretrained", args.pretrained, "--output", teacher,
                "--devices", args.teacher_devices, "--workers", 4, "--seed", 43,
                *(["--resume", resume] if resume.exists() else [])], teacher / "COMPLETE.json")
            second = analyze_seed(43)
            update(state="complete", phase="P1_confirmation_ready", seed42_gate=first, seed43_gate=second,
                   next_action="P2 remains gated; interpret both fixed-seed results before the eight-group training matrix")
        else:
            update(state="complete", phase="P1_seed42_gate_ready", seed42_gate=first,
                   next_action=first["next_action"], confirmation_requested=args.confirm_if_pass)
    except Exception as exc:
        update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
