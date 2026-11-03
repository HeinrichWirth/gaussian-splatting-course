import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from reconstruction import (finite_render, orbit, small_scene, synthetic_capture, fit_colors,
                            color_basis, read_capture, write_capture, evaluate, scores, triangulate)
from reconstruction_demos import photo_capture
from splat_render import render_scene


class ReconstructionTests(unittest.TestCase):
    def test_tiled_matches_same_support_at_rotated_image_border(self):
        sc=small_scene()
        sc["splats"][0].update(center=[-1.6,.4,3.5],rotation=[.82,.15,.2,.5])
        for side,k in [(3,1.1),(8,3.),(64,4.)]:
            img,stats=finite_render(sc,orbit(18,37,29),37,29,side,k)
            brute,counts=finite_render(sc,orbit(18,37,29),37,29,side,k,False)
            np.testing.assert_allclose(img,brute,atol=1e-13,rtol=0)
            self.assertLessEqual(stats["pairs"],counts["pairs"])
        full=render_scene(sc,orbit(18,37,29),37,29)
        self.assertGreater(np.max(np.abs(full-img)),0)

    def test_stereo_degenerate_and_noisy(self):
        self.assertIsNone(triangulate(80,0,40,40))
        self.assertIsNone(triangulate(80,.4,40,41))
        self.assertAlmostEqual(triangulate(80,.4,40,33),32/7)

    def test_saved_capture_and_held_out_rgb(self):
        fs,initial=synthetic_capture()
        with tempfile.TemporaryDirectory() as tmp:
            write_capture(tmp,fs,initial)
            disk,scene=read_capture(tmp)
            self.assertEqual(initial,scene)
            for a,b in zip(fs,disk):
                np.testing.assert_array_equal(a["rgb"],b["rgb"])
                self.assertEqual(a["camera"],b["camera"])
        trained,loss=fit_colors(initial,fs,12)
        other=deepcopy(fs)
        for f in other:
            if f["split"]=="test": f["rgb"][:]=0
        again,_=fit_colors(initial,other,12)
        self.assertEqual(trained,again)
        self.assertTrue(all(a>=b-1e-14 for a,b in zip(loss,loss[1:])))
        self.assertLess(evaluate(trained,fs)[0]["mse"],evaluate(initial,fs)[0]["mse"])

    def test_weights_predict_rgb(self):
        fs,sc=synthetic_capture()
        f=fs[0];A,T=color_basis(sc,f)
        h,w=f["rgb"].shape[:2]
        np.testing.assert_allclose(A.sum(1)+T[:,0],1)
        np.testing.assert_allclose(A@np.array([p["color"] for p in sc["splats"]])+T,
                                   render_scene(sc,f["camera"],w,h).reshape(-1,3))

    def test_photo_calibration_and_fixed_split(self):
        frames,scene=photo_capture()
        self.assertEqual([f["name"] for f in frames if f["split"]=="test"],["view-04.jpg"])
        self.assertEqual(len(scene["splats"]),24)
        self.assertEqual(frames[0]["rgb"].shape,(30,40,3))
        self.assertEqual(frames[0]["camera"]["principal"],[19.5,14.5])
        self.assertTrue(all(p["color"]==[.5]*3 for p in scene["splats"]))

    def test_differentiable_renderer_matches_training_and_export(self):
        from reconstruction_fit import refine_scene
        import torch
        torch.set_num_threads(1)
        fs,sc=synthetic_capture()
        sc["splats"][0]["center"][0]-=.05
        sc["splats"][1]["rotation"]=[.97,.14,.1,-.05]
        model,history=refine_scene(sc,fs,steps=2,rate=.005)
        before=np.mean([r["mse"] for r in evaluate(sc,fs,"train")])
        after=np.mean([r["mse"] for r in evaluate(model,fs,"train")])
        self.assertAlmostEqual(history[0],before,places=10)
        self.assertAlmostEqual(history[-1],after,places=10)
        self.assertLess(after,before)

    def test_metric_units(self):
        x=np.array([.2,.4,.6,.8]); y=np.array([.3,.4,.4,.9])
        result=scores(y,x)
        self.assertAlmostEqual(result["mse"],.015)
        self.assertAlmostEqual(result["psnr"],18.2390874,places=6)
        self.assertEqual(scores(x,x)["psnr"],math.inf)
