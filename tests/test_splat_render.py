import importlib.util
import unittest

import numpy as np

import splat_render as gs


class SplatRenderTests(unittest.TestCase):
    def test_depth_order_pixel_and_near_plane(self):
        scene, cam = gs.overlap_scene(), gs.orbit_camera()
        self.assertEqual([s["index"] for s in gs.project_splats(scene,cam,65,65)], [0,1])
        np.testing.assert_allclose(gs.render_scene(scene,cam)[32,32], [.8,.2,.4])
        np.testing.assert_allclose(gs.render_scene(scene,cam,reverse=True)[32,32], [.5,.2,.7])
        self.assertEqual([s["index"] for s in gs.project_splats(scene,gs.orbit_camera(135),65,65)], [1,0])
        scene["splats"][0]["center"][2]=-.1
        self.assertEqual([s["index"] for s in gs.project_splats(scene,cam,65,65)], [1])

    def test_sh_basis_direction_and_odd_degree_signs(self):
        np.testing.assert_allclose(gs.sh_basis([3,0,4]), [gs.C0,0,.8*gs.C1,-.6*gs.C1])
        coeff=[[0,0,0],[0,0,0],[.4,0,0],[.6,0,0]]
        positive=gs.sh_color([3,0,4],coeff)
        np.testing.assert_allclose(positive,[.480455899524,.5,.5])
        np.testing.assert_allclose(positive+gs.sh_color([-3,0,-4],coeff),np.ones(3))
        with self.assertRaises(ValueError):
            gs.sh_color([0,0,0],coeff)

    def test_scale_depth_ambiguity_and_held_out_view(self):
        cam=gs.translated_camera(0)
        np.testing.assert_allclose(gs.render_scene(gs.ray_scene(2.4),cam),
                                   gs.render_scene(gs.ray_scene(4),cam),atol=1e-14)
        next_z, slope=gs.depth_step(2.4)
        self.assertAlmostEqual(slope,-.0041669008184,places=10)
        self.assertAlmostEqual(next_z,2.7333520655,places=8)
        self.assertLess(gs.ray_loss(next_z,(-.6,)),gs.ray_loss(2.4,(-.6,)))
        self.assertAlmostEqual(gs.depth_step(2.4,(0,))[0],2.4)

    def test_density_transitions_preserve_input_and_report_actual_loss(self):
        start=gs.density_scene()
        expected={'clone':(3,.01214659),'split':(3,.00347211),'prune':(1,.00467882),'reset':(2,.01592582)}
        for op,(count,loss) in expected.items():
            result=gs.density_edit(start,op)
            self.assertEqual(len(result["splats"]),count)
            self.assertAlmostEqual(gs.density_loss(result),loss,places=7)
        self.assertEqual(start,gs.density_scene())
        empty=gs.density_edit(gs.density_edit(start,'reset'),'prune')
        np.testing.assert_allclose(gs.render_scene(empty),1)
        adaptive=gs.density_pass(start)
        self.assertEqual(len(adaptive["splats"]),2)
        self.assertLess(gs.density_loss(adaptive),gs.density_loss(start))
        self.assertAlmostEqual(adaptive["splats"][0]["center"][0],-.5695)

    @unittest.skipIf(importlib.util.find_spec('torch') is None, 'requires requirements-training.txt')
    def test_differentiable_render_matches_reference_with_nonzero_covariance_xy(self):
        import torch
        from splat_fit import render_parallel
        scene=gs.density_scene()
        scene["splats"][0]["center"]=[.7,.5,3.8]
        dtype=torch.float64
        args=[torch.tensor([s[key] for s in scene['splats']],dtype=dtype)
              for key in ('center','scales','color','opacity')]
        args[0].requires_grad_()
        img=render_parallel(*args,.6)
        np.testing.assert_allclose(img.detach().numpy(),gs.render_scene(scene,gs.translated_camera(.6),33,33),atol=1e-12)
        value=img[18,15,0]
        value.backward()
        def sample(offset):
            scene['splats'][0]['center'][0]=.7+offset
            return gs.render_scene(scene,gs.translated_camera(.6),33,33)[18,15,0]
        h=1e-5
        self.assertAlmostEqual(args[0].grad[0,0].item(),(sample(h)-sample(-h))/(2*h),places=7)
