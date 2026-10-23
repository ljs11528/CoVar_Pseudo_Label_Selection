"""Synthetic end-to-end screen fixture; always marked synthetic, no real fitting."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .anchor_trust import apply_model
from .bounded_probe import image_folds
from .history_competition import current_features
from .prepare_data import digest
from .run_history_screen import run
from .run_score_audit import write_json


def fixture(root, spec):
    package = Path(__file__).resolve().parent
    ids = {role:[f"{role}_image{i:03d}" for i in range(count)] for role,count in spec["role_counts"].items()}
    raw = {}
    for role, names in ids.items():
        for i, name in enumerate(names):
            seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
            rng = np.random.default_rng(seed)
            n, classes = 64, spec["classes"]
            z = rng.normal(size=(n, classes)).astype(np.float32)
            a = rng.integers(classes, size=n); z[np.arange(n), a] = 10.
            current = current_features(z)
            gt = a.copy(); gt[::4] = current["b"][::4]
            pixels = rng.permutation(n)
            raw[name] = dict(z=z, GT=gt.astype(np.int64), pixel_indices=pixels,
                yx=np.column_stack([pixels//8, pixels % 8]), original_size=np.array([8, 8]),
                valid_pixel_count=np.array(n), boundary=np.arange(n) % 3 == 0,
                role=np.asarray(role), image_id=np.asarray(name))
    for stage, scale in zip(spec["input_exports"], (.8, .9, 1.)):
        folder = root/"experiments"/spec["input_exports"][stage]; folder.mkdir(parents=True)
        signature = dict(stage=stage, seed=42, protocol_sha256=digest(package/"protocol.json"),
            checkpoint_sha256=hashlib.sha256(stage.encode()).hexdigest(), partition_receipt_sha256="f"*64)
        np.savez_compressed(folder/"prototypes.npz", image_ids=np.asarray(["sup0", "sup1"]))
        prototype_hash = digest(folder/"prototypes.npz")
        entries = []
        for role, names in ids.items():
            for name in names:
                row = raw[name]; z = row["z"]*scale
                path = folder/"samples"/role/(name+".npz"); path.parent.mkdir(parents=True, exist_ok=True)
                arrays = {k:v for k,v in row.items() if k != "z"}
                arrays.update(logits=z, correct=z.argmax(1) == row["GT"], score__pseudo_label=z.argmax(1),
                    score__strict_conflict=np.arange(len(z)) % 7 == 0,
                    checkpoint_sha256=np.asarray(signature["checkpoint_sha256"]),
                    protocol_sha256=np.asarray(signature["protocol_sha256"]), prototype_sha256=np.asarray(prototype_hash))
                np.savez_compressed(path, **arrays)
                entries.append(dict(role=role, image_id=name, file=str(path.relative_to(folder)),
                                    bytes=path.stat().st_size, sha256=digest(path)))
        write_json(folder/"manifest.json", dict(synthetic=True, signature=signature, files=entries, prototype_sha256=prototype_hash))
        write_json(folder/"COMPLETE.json", dict(state="complete", signature=signature, manifest_sha256=digest(folder/"manifest.json")))
    source = root/spec["ranking"]["saved_audit"]; source.mkdir(parents=True)
    stage_receipts = {}
    for stage, scale in (("middle", .9), ("late", 1.)):
        folder = source/("seed42_"+stage); folder.mkdir()
        mc = dict(kind="MC", mc=dict(a=1., b=0., mean=0., scale=1.))
        mm = dict(kind="MM", parent=mc, coefficients=[0., .1], basis=dict(mean=[0., 0.], scale=[1., 1.],
            columns=[dict(kind="linear", column=1, center=0., scale=1.)], intercept=True))
        models = dict(MC=mc, MM=mm)
        write_json(folder/"models.json", models)
        write_json(folder/"outer_models.json", {f"fold{f}__{name}":model for f in range(3) for name,model in models.items()})
        for role, names in ids.items():
            pieces = []
            for i, name in enumerate(names):
                row = raw[name]; n = min(spec["cap"], len(row["GT"]))
                piece = current_features(row["z"][:n]*scale)
                piece.update(GT=row["GT"][:n], pixel_indices=row["pixel_indices"][:n], image=np.full(n, i))
                pieces.append(piece)
            pool = {k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
            pool.update(role=role, image_ids=names)
            pool["error"] = (pool["pseudo_label"] != pool["GT"]).astype(np.int8)
            np.savez_compressed(folder/(role+"_pool.npz"), **{k:np.asarray(v) for k,v in pool.items()})
            np.savez_compressed(folder/(role+"_predictions.npz"), **{name:apply_model(m,pool) for name,m in models.items()})
            if role == "D_rel":
                folds = image_folds(names, pool["image"], 3, spec["folds"]["namespace"])
                write_json(folder/"folds.json", [dict(heldout_images=[names[i] for i in np.unique(pool["image"][b])],
                    train_images=[names[i] for i in np.unique(pool["image"][a])]) for a,b in folds])
        write_json(folder/"COMPLETE.json", dict(state="complete", synthetic=True,
            signature=dict(source_commit=spec["ranking"]["source_commit"]),
            files={f.name:dict(bytes=f.stat().st_size, sha256=digest(f)) for f in folder.iterdir() if f.is_file()}))
        stage_receipts[stage] = digest(folder/"COMPLETE.json")
    write_json(source/"COMPLETE.json", dict(state="complete", synthetic=True, stage_receipts=stage_receipts))


def smoke(output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    original = json.loads(Path(__file__).with_name("history_protocol.json").read_text())
    spec = copy.deepcopy(original)
    spec["role_counts"] = dict(D_rel=12, validation=9)
    spec["cap"] = 48
    spec["screen"]["bootstrap_replicates"] = 60
    spec["screen"]["minimum_support"] = dict(error_images=1, correct_images=1, error_pixels=1, correct_pixels=1)
    root = output/"fixture"; fixture(root, spec)
    complete = run(root, output/"run", spec, synthetic=True)
    assert complete["state"] == "complete" and complete["new_fits"] == 0
    gate = json.loads((output/"run/screen_gate.json").read_text())
    assert not gate["passed"]
    assert (output/"run/PROBES_SKIPPED.json").exists()
    for role in spec["role_counts"]:
        check = json.loads((output/"run"/(role+"_alignment.json")).read_text())
        assert check["images"] == spec["role_counts"][role] and check["all_full_pixel_arrays_equal"]
        assert check["full_pixel_positions"] > check["analyzed_pixel_positions"]
        for pair in spec["pairs"]:
            replay = json.loads((output/"run"/pair["name"]/(role+"_baseline_replay.json")).read_text())
            assert max(replay["max_abs_difference"].values()) <= 1e-10
    write_json(output/"SMOKE_COMPLETE.json", dict(state="passed", synthetic=True, new_fits=0,
        screen_failed_as_constructed=True, real_protocol_unchanged=original == json.loads(Path(__file__).with_name("history_protocol.json").read_text()),
        output_files=len(complete["files"]), receipt_sha256=digest(output/"run/COMPLETE.json")))
    return output/"SMOKE_COMPLETE.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(smoke(args.output))
