import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from ..anchor_trust import (apply_model, candidate_sets, design, fit_basis64, fit_candidates,
                            geometry, newton, select, shuffle_tuples)
from ..bounded_probe import logit_metrics
from ..anchor_trust_analysis import evaluate, folds_for


def spec():
    return json.loads(Path(__file__).resolve().parents[1].joinpath("anchor_trust_protocol.json").read_text())


def synthetic_pool(images=30,pixels=24,seed=905):
    rng=np.random.default_rng(seed)
    n=images*pixels;classes=21
    z=rng.normal(size=(n,classes))*2
    z[np.arange(n),rng.integers(classes,size=n)]+=rng.uniform(2,12,n)
    s=rng.uniform(-.8,.8,size=z.shape)
    counts=np.arange(1,classes+1)*23
    real=geometry(z,s,counts)
    pcorrect=1/(1+np.exp(-(.25*real["logit_mc"]+.4*real["features"][:,3])))
    error=(rng.random(n)>pcorrect).astype(np.int8)
    gt=np.where(error,(real["a"]+rng.integers(1,classes,n))%classes,real["a"])
    pool={k:real[k] for k in ("mc","p2","margin","entropy","logit_mc","strict_conflict","b","count_a","count_b","count_c")}
    pool.update(role="D_rel",image_ids=[f"synth_{i}" for i in range(images)],image=np.repeat(np.arange(images),pixels),
                error=error,GT=gt,pseudo_label=real["a"],prototype_label=real["c"],boundary=rng.integers(2,size=n),
                sampling_weight=rng.uniform(.8,2,size=n),pixel_indices=np.tile(np.arange(pixels),images),
                prototype_counts=counts,anchor__real=real["features"])
    for seed in spec()["permutation"]["seeds"]:
        pieces=[]
        for i in range(images):
            ix=np.arange(i*pixels,(i+1)*pixels)
            ss,nn=shuffle_tuples(z[ix],s[ix],counts,pool["image_ids"][i],pool["pixel_indices"][ix],seed)
            sham=geometry(z[ix],ss,nn)["features"]
            sham[:,7]=real["features"][ix,7];sham[:,9]=sham[:,3]*sham[:,7]
            pieces.append(sham)
        pool[f"anchor__sham{seed}"]=np.concatenate(pieces)
    return pool


class AnchorTrustTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_probability_identity_extremes(self):
        z=np.array([[1000.,0.,-1000.],[0,0,0],[np.log(.5),np.log(.4),np.log(.1)]])
        result=geometry(z,np.zeros_like(z),np.array([10,20,30]))
        np.testing.assert_allclose(result["p2"],result["mc"]-result["margin"],atol=1e-15)
        self.assertTrue(np.isfinite(result["features"]).all())
        self.assertTrue(np.isfinite(result["logit_mc"]).all())

    def test_tuple_shuffle_identity_and_batch_invariance(self):
        rng=np.random.default_rng(3)
        z=rng.normal(size=(20,7));s=rng.uniform(-1,1,(20,7));counts=np.arange(7)*11+1
        pixels=np.arange(200,220)
        ss,nn=shuffle_tuples(z,s,counts,"sample",pixels,301)
        sa,na=shuffle_tuples(z[:9],s[:9],counts,"sample",pixels[:9],301)
        sb,nb=shuffle_tuples(z[9:],s[9:],counts,"sample",pixels[9:],301)
        np.testing.assert_array_equal(ss,np.concatenate([sa,sb]))
        np.testing.assert_array_equal(nn,np.concatenate([na,nb]))
        a=z.argmax(1)
        np.testing.assert_array_equal(s[np.arange(20),a],ss[np.arange(20),a])
        np.testing.assert_array_equal(np.sort(s,1),np.sort(ss,1))
        for i in range(20):
            self.assertEqual(sorted(zip(s[i],counts)),sorted(zip(ss[i],nn[i])))
        real,sham=geometry(z,s,counts),geometry(z,ss,nn)
        np.testing.assert_allclose(real["features"][:,[0,2,5,7]],sham["features"][:,[0,2,5,7]],atol=1e-14)
        self.assertGreater(np.max(np.abs(real["features"][:,3]-sham["features"][:,3])),.01)

    def test_basis_deduplicates_sign_and_train_only(self):
        x=np.arange(50,dtype=float)
        raw=np.column_stack([x,-x,np.zeros(50)])
        basis=fit_basis64(raw,np.ones(50),"linear")
        self.assertEqual(len(basis["columns"]),1)
        saved=json.dumps(basis)
        design(basis,np.array([[10000.,-10000.,9.]]))
        self.assertEqual(saved,json.dumps(basis))

    def test_newton_against_independent_optimizer(self):
        from scipy.optimize import minimize
        rng=np.random.default_rng(17)
        x=np.column_stack([np.ones(300),rng.normal(size=(300,4))])
        offset=rng.normal(size=300);y=rng.integers(2,size=300);w=rng.uniform(.2,2,300);w/=w.sum()
        cfg=spec()["optimizer"]
        actual,info=newton(x,offset,y,w,cfg,"cpu")
        def objective(beta):
            l=offset+x@beta
            return np.sum(w*(np.logaddexp(0,l)-y*l))+cfg["l2"]/2*(beta@beta)
        def jac(beta):
            l=offset+x@beta
            return x.T@(w*(1/(1+np.exp(-l))-y))+cfg["l2"]*beta
        ref=minimize(objective,np.zeros(5),jac=jac,method="BFGS",options={"gtol":1e-10})
        np.testing.assert_allclose(actual,ref.x,atol=1e-5,rtol=1e-5)
        self.assertTrue(info["converged"])

    def test_conflict_only_serialization_and_role_guard(self):
        pool=synthetic_pool(6,12)
        cfg=spec();cfg["permutation"]["seeds"]=[]
        train=np.flatnonzero(pool["image"]<4);held=np.flatnonzero(pool["image"]>=4)
        models=fit_candidates(pool,train,held,cfg,"cpu",{},lambda row:None)
        for name in ("real_G","real_S","real_T"):
            output=apply_model(models[name],pool,held)
            np.testing.assert_array_equal(output,apply_model(json.loads(json.dumps(models[name])),pool,held))
            nonconf=~pool["strict_conflict"][held]
            np.testing.assert_array_equal(output[nonconf],apply_model(models["MM_additive"],pool,held)[nonconf])
        forbidden=dict(pool,role="validation")
        with self.assertRaises(ValueError):
            fit_candidates(forbidden,train,held,cfg,"cpu",{},lambda row:None)
        with self.assertRaises(ValueError):
            fit_candidates(pool,train,train[:5],cfg,"cpu",{},lambda row:None)

    def test_zero_and_shared_family_selection(self):
        families=candidate_sets(spec())
        losses={name:1. for names in families.values() for name in names}
        for names in families.values():
            self.assertEqual(select(losses,names,1e-5),"MC")
        losses["real_S"]-=.01
        self.assertEqual(select(losses,families["real_T"],1e-5),"real_S")
        self.assertEqual(select(losses,families["sham301_T"],1e-5),"MC")

    def test_fold_metric_invariant_to_fold_specific_offsets(self):
        pool=synthetic_pool(12,12)
        cfg=spec()
        clean=1.-2*pool["error"].astype(float)
        shifted=clean.copy()
        for f,(_,held) in enumerate(folds_for(pool,cfg)):
            shifted[held]+=f*100
        rows,curves=evaluate(pool,{"MC":clean,"MM":shifted},cfg)
        def value(method,fold):
            return next(r["error_ap"] for r in rows if r["scope"]=="all" and r["method"]==method and r["fold"]==fold)
        self.assertAlmostEqual(value("MC","mean"),value("MM","mean"))
        self.assertGreater(value("MC","pooled"),value("MM","pooled"))
        self.assertTrue(all(np.isfinite(r["risk"]) for r in curves))


if __name__=="__main__":
    unittest.main()
