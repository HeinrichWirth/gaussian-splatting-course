"""Check the deformation decoder, stereo fit and portable motion schema."""
from copy import deepcopy
import json
import unittest

import numpy as np

import deformation as motion
from splat_render import render_scene, orbit_camera


class DeformationTests(unittest.TestCase):
    def test_field_interpolation_updates_three_static_attributes(self):
        self.assertEqual(motion.field_scene(0), motion.ribbon())
        self.assertAlmostEqual(motion.field_value(.9,.5), .45)
        positive = motion.field_scene(.5)["splats"][-1]
        negative = motion.field_scene(.5,-1)["splats"][-1]
        np.testing.assert_allclose(positive["center"], [.8,-.36,4])
        np.testing.assert_allclose(negative["scales"][0], .23355)
        self.assertAlmostEqual(np.linalg.norm(positive["rotation"]),1)

    def test_heldout_center_and_rendered_frame(self):
        rows = motion.stereo_tracks()
        self.assertEqual([r["t"] for r in rows], [.25,.75])
        model = motion.fit_motion(motion.ribbon(), rows)
        np.testing.assert_allclose(model["amplitudes"][-1],[.12,-.8,.45])
        scene = motion.scene_at(json.loads(json.dumps(model)), .375)
        np.testing.assert_allclose(scene["splats"][-1]["center"], [.9125,-.75,4.421875])
        cam = orbit_camera(32,64)
        predicted = render_scene(scene,cam,65,49)
        actual = render_scene(motion.reference_scene(.375),cam,65,49)
        self.assertAlmostEqual(np.mean((predicted-actual)**2), .00036425478483558)
        noisy = motion.fit_motion(motion.ribbon(), motion.stereo_tracks(1))
        self.assertFalse(np.allclose(model["amplitudes"],noisy["amplitudes"]))

    def test_duration_bounds_and_bad_observations(self):
        model = motion.example_model()
        longer = deepcopy(model)
        longer["duration"] = 2.
        self.assertEqual(motion.scene_at(model,.5), motion.scene_at(longer,1))
        self.assertEqual(motion.scene_at(longer,2), longer["canonical"])
        with self.assertRaises(ValueError):
            motion.scene_at(longer,2.1)
        with self.assertRaises(ValueError):
            motion.validate_model({**model, "amplitudes": [[0,0,float("nan")]]*5})
        row = motion.stereo_tracks()[0]
        with self.assertRaises(ValueError):
            motion.fit_motion(motion.ribbon(), [{**row,"t":0}])
        with self.assertRaises(ValueError):
            motion.fit_motion(motion.ribbon(), [{**row,"right":row["left"]}])
