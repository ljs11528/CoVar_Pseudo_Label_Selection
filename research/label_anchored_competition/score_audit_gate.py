"""Apply only the pre-registered D_rel resource gate; validation is inadmissible."""
import math


def decide(spec, intervals, stability):
    if any(row.get("role", "D_rel") != "D_rel" for row in intervals+stability):
        raise ValueError("Validation cannot decide the bounded-audit resource gate")
    pairs=[tuple(p) for p in spec["statistics"]["primary_contrasts"]]
    checks=[]
    def record(name, passed, details):
        checks.append(dict(condition=name,passed=bool(passed),details=details))
    def find(stage,scope,a,b):
        matches=[r for r in intervals if (r.get("stage"),r.get("scope"),r.get("method"),r.get("comparator"))==(stage,scope,a,b)]
        return matches[0] if len(matches)==1 else None
    for a,b in pairs+[("e","MC")]:
        row=find("late","conflict",a,b)
        level=spec["statistics"]["primary_confidence"] if (a,b) in pairs else .95
        valid=(row is not None and row.get("adequate_support") in (True,"True")
               and math.isclose(float(row["confidence"]),level,abs_tol=1e-12)
               and int(row["valid_replicates"])==spec["statistics"]["bootstrap_replicates"])
        record(f"late_conflict_{a}_minus_{b}_lower_bound",valid and float(row["ci_low"])>0,
               row or "missing or duplicated interval")
    for a,b in pairs:
        rows=[find(stage,"conflict",a,b) for stage in spec["stages"]]
        positive=sum(r is not None and float(r["delta_error_ap"])>0 for r in rows)
        record(f"two_of_three_stages_{a}_minus_{b}",positive>=2,dict(positive_stages=positive))
        rows=[r for r in stability if r["stage"]=="late" and r["scope"]=="conflict" and
              (r["method"],r["comparator"])==(a,b)]
        expected={(s,u) for s in spec["correction"]["fit_seeds"] for u in spec["correction"]["budgets"]}
        observed={(int(r["seed"]),int(r["budget"])) for r in rows}
        record(f"all_fixed_budgets_and_seeds_{a}_minus_{b}",observed==expected and len(rows)==len(expected)
               and all(float(r["delta_error_ap"])>0 for r in rows),rows)
    for stage in spec["stages"]:
        for a,b in (("e","MC"),("controls_e","controls")):
            row=find(stage,"all",a,b)
            valid=(row is not None and row.get("adequate_support") in (True,"True")
                   and int(row["valid_replicates"])==spec["statistics"]["bootstrap_replicates"])
            record(f"no_clear_global_harm_{stage}_{a}_minus_{b}",valid and float(row["ci_high"])>=0,
                   row or "missing interval")
    passed=all(row["passed"] for row in checks)
    return dict(eligible_for_one_bn_confirmation_teacher=passed,uses_validation=False,
        decision="one repaired seed42 teacher eligible after real multi-GPU save/reload smoke" if passed
                 else "stop expansion of this residual-weighted formula; no confirmation teacher authorized by this gate",
        checks=checks,limitations="Post-hoc development on one existing old-BN seed42 trajectory; not a P2 or generalization gate")
