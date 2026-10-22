import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from research.label_anchored_competition.bounded_probe import (
    apply_probe, choose_candidate, fit_basis, fit_correction_path, fit_mc,
    image_folds, logit_metrics, mc_logits, sigmoid, transform_basis)
from research.label_anchored_competition.run_score_audit import key, nested_fit
from research.label_anchored_competition.score_audit_diagnostics import weighted_correlation, weighted_midrank


def specification():
    return json.loads((Path(__file__).parents[1]/"score_audit_protocol.json").read_text())


class BoundedProbeInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mc_optimizer_and_monotonicity(self):
        from scipy.optimize import minimize
        rng = np.random.default_rng(82)
        cfg = specification()["mc_calibrator"]
        for slope in (1.3, -2., 0.):
            x = rng.normal(size=600)*4
            y = rng.binomial(1, sigmoid(slope*x+.5))
            w = rng.uniform(.1, 4, len(x))
            model = fit_mc(x, y, w, cfg)
            z = (x-model["mean"])/model["scale"]
            def loss(t):
                return logit_metrics(y, t[0]*z+t[1], w)["nll"]+cfg["l2"]*.5*np.dot(t,t)
            ref = minimize(loss, [1.,0.], method="L-BFGS-B", bounds=[(1e-6,None),(None,None)],
                           options=dict(ftol=1e-14, gtol=1e-9))
            self.assertAlmostEqual(model["objective"], ref.fun, places=10)
            self.assertTrue((np.diff(mc_logits(model, np.sort(x))) > 0).all())
        for x,y in [(np.full(100, 2.), np.ones(100)), (np.linspace(-60,60,100), np.zeros(100))]:
            model = fit_mc(x,y,np.ones(100),cfg)
            self.assertTrue(np.isfinite(mc_logits(model,x)).all())

    def test_role_guards_and_zero_identity(self):
        cfg = specification()
        for role in ("validation", "D_U"):
            with self.assertRaises(ValueError):
                fit_mc(np.arange(20), np.ones(20), np.ones(20), cfg["mc_calibrator"], role)
            with self.assertRaises(ValueError):
                fit_basis(np.ones((20,1)), np.ones(20), "linear", [.5], role)
            with self.assertRaises(ValueError):
                nested_fit(dict(role=role), {}, cfg, None, "cpu", None)
        x=np.linspace(-4,4,40)
        mc=fit_mc(x,np.tile([0,1],20),np.ones(40),cfg["mc_calibrator"])
        np.testing.assert_array_equal(apply_probe(dict(mc=mc,form="zero"), None, x), mc_logits(mc,x))
        self.assertEqual(choose_candidate(dict(zero=.2, linear=.2-5e-7, additive=.2), 1e-6), "zero")

    def test_folds_are_image_isolated_and_identity_stable(self):
        ids = [f"img{i}" for i in range(18)]
        image=np.repeat(np.arange(18),3)
        folds=image_folds(ids,image,3,"test")
        seen=[]
        for train,held in folds:
            self.assertFalse(set(image[train]) & set(image[held]))
            seen.extend(held.tolist())
        self.assertEqual(sorted(seen), list(range(len(image))))
        reverse=image_folds(ids[::-1],17-image,3,"test")
        for a,b in zip(folds,reverse):
            np.testing.assert_array_equal(a[1],b[1])

    def test_constant_basis_and_prefix(self):
        cfg=specification()["correction"]
        cfg.update(budgets=[4,12],curve_updates=[0,4,12],batch_pixels=32)
        rng=np.random.default_rng(12)
        x=rng.normal(size=(100,2)).astype(np.float32)
        w=np.ones(100); offset=np.zeros(100); y=(x[:,0]>0).astype(float)
        long=fit_correction_path(x,offset,y,w,x,offset,y,w,cfg,22,"cpu")
        short=fit_correction_path(x,offset,y,w,x,offset,y,w,dict(cfg,budgets=[4]),22,"cpu")
        np.testing.assert_array_equal(long[4]["coefficients"],short[4]["coefficients"])
        self.assertLess(logit_metrics(y,long[12]["heldout_logits"],w)["nll"],np.log(2))
        basis=fit_basis(np.ones((100,2)), w,"additive",[.25,.5,.75])
        empty=transform_basis(basis,np.ones((100,2)))
        self.assertEqual(empty.shape,(100,0))
        zero=fit_correction_path(empty,offset,y,w,empty,offset,y,w,cfg,22,"cpu")
        np.testing.assert_array_equal(zero[12]["heldout_logits"],offset)

    def test_weighted_midrank_ties(self):
        x=np.array([1.,1.,3.,2.]); w=np.array([1.,3.,2.,4.])
        np.testing.assert_allclose(weighted_midrank(x,w),[.2,.2,.9,.6])
        self.assertAlmostEqual(weighted_correlation(x,x,w),1.)

    def test_nested_pipeline_round_trip_and_frozen_mc(self):
        spec=specification()
        spec["correction"].update(budgets=[4,12], curve_updates=[0,4,12], batch_pixels=32)
        rng=np.random.default_rng(71)
        n=18*20
        lm=rng.normal(1,1,n); e=rng.normal(size=n)
        error=1-rng.binomial(1,sigmoid(lm-1.2*e))
        pool=dict(role="D_rel",image_ids=[f"img{i}" for i in range(18)],image=np.repeat(np.arange(18),20),
                  error=error, sampling_weight=rng.uniform(1,4,n),logit_mc=lm)
        x={m:rng.normal(size=(n,len(columns))) for m,columns in spec["feature_sets"].items()}
        x["e"][:,0]=e
        # A constant feature must select zero, exercising exact fallback inside outer CV.
        x["uniform"][:]=1
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            models,oof=nested_fit(pool,x,spec,output,"cpu",lambda *a,**kw:None)
            serialized=json.loads((output/"models.json").read_text())
            for name,model in models.items():
                self.assertTrue(np.isfinite(oof[name]).all())
                features=x[model["feature_set"]] if name!="MC" else None
                np.testing.assert_array_equal(apply_probe(model,features,lm),apply_probe(serialized[name],features,lm))
                self.assertEqual(model["mc"],models["MC"]["mc"])
            for seed in spec["correction"]["fit_seeds"]:
                for budget in spec["correction"]["budgets"]:
                    name=key(seed,budget,"uniform")
                    self.assertEqual(models[name]["form"],"zero")
                    np.testing.assert_array_equal(oof[name],oof["MC"])
            outer=json.loads((output/"outer_models.json").read_text())
            self.assertEqual(len(outer),72)
            for row in json.loads((output/"folds.json").read_text()):
                for inner in row["inner"]:
                    self.assertFalse(set(inner["train_images"]) & set(row["heldout_images"]))


if __name__ == "__main__":
    unittest.main()
