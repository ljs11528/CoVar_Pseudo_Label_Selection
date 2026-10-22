import copy
import unittest
from research.label_anchored_competition.score_audit_gate import decide
from research.label_anchored_competition.tests.test_bounded_probe import specification


def positive_fixture():
    spec=specification()
    rows=[]; stability=[]
    for stage in spec["stages"]:
        for scope in ("all","conflict"):
            for a,b in spec["statistics"]["primary_contrasts"]+[["e","MC"]]:
                rows.append(dict(stage=stage,role="D_rel",scope=scope,method=a,comparator=b,
                    delta_error_ap=.02,ci_low=.01,ci_high=.03,adequate_support=True,valid_replicates=2000,
                    confidence=.9875 if scope=="conflict" and b!="MC" else .95))
        for seed in spec["correction"]["fit_seeds"]:
            for budget in spec["correction"]["budgets"]:
                for a,b in spec["statistics"]["primary_contrasts"]:
                    stability.append(dict(stage=stage,role="D_rel",scope="conflict",method=a,comparator=b,
                        seed=seed,budget=budget,delta_error_ap=.02))
    return spec,rows,stability


class BoundedGate(unittest.TestCase):
    def test_all_fixed_conditions_only_allow_one_teacher(self):
        result=decide(*positive_fixture())
        self.assertTrue(result["eligible_for_one_bn_confirmation_teacher"])
        self.assertFalse(result["uses_validation"])

    def test_zero_lower_bound_missing_interval_or_sampler_reversal_stops(self):
        for condition in ("zero", "missing", "reversal", "harm"):
            spec,rows,stability=positive_fixture()
            if condition=="zero":
                next(r for r in rows if r["stage"]=="late" and r["scope"]=="conflict")["ci_low"]=0
            if condition=="missing":
                rows=rows[:-1]
            if condition=="reversal":
                stability[-1]["delta_error_ap"]=-.001
            if condition=="harm":
                next(r for r in rows if r["stage"]=="early" and r["scope"]=="all" and r["method"]=="e" and r["comparator"]=="MC")["ci_high"]=-.001
            self.assertFalse(decide(spec,rows,stability)["eligible_for_one_bn_confirmation_teacher"])

    def test_validation_cannot_enter_gate(self):
        spec,rows,stability=positive_fixture()
        stability[0]["role"]="validation"
        with self.assertRaises(ValueError):
            decide(spec,rows,stability)


if __name__=="__main__":
    unittest.main()
