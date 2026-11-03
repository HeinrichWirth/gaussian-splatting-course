import unittest

import numpy as np

import dynamic_motion as motion
from splat_render import project_splats, make_scene, translated_camera


class MotionTests(unittest.TestCase):
    def test_same_orange_drift_but_different_stationary_landmark(self):
        a = motion.observing_pair(object_speed=1)[0][0]
        b = motion.observing_pair(object_speed=0, camera_speed=-1)[0][0]
        self.assertEqual((a['u'], b['u'], a['anchor'], b['anchor']), (64, 64, 24, 40))
        views, disparity, depth = motion.observing_pair(lag=.2)
        self.assertAlmostEqual(disparity, 12)
        self.assertAlmostEqual(depth, 16/3)
        self.assertIsNone(motion.observing_pair(lag=.8)[2])

    def test_occluder_blocks_only_one_center_and_masks_its_pixels(self):
        views, _, depth = motion.observing_pair(gate=True)
        self.assertEqual([v['visible'] for v in views], [False, True])
        self.assertIsNone(depth)
        for view in views:
            u = round(view['u'])
            plain = motion.observation_image(view)[30, u]
            with_gate = motion.observation_image(view, gate=True)[30, u]
            self.assertTrue(plain[0] > plain[2])
            if view['visible']:
                np.testing.assert_array_equal(plain, with_gate)
            else:
                np.testing.assert_array_equal(with_gate, [.23, .29, .30])

    def test_fit_uses_only_observations_and_extrapolates_their_error(self):
        times = np.array([0., .5, 1., 2., 3.])
        np.testing.assert_allclose(motion.fit_path(times, True), motion.true_x(times))
        error = motion.fit_path(times, True, .1) - motion.true_x(times)
        np.testing.assert_allclose(error, [0, .075, .1, 0, -.3], atol=1e-14)
        self.assertAlmostEqual(motion.fit_path(1), .1)
        spot = motion.trajectory_splat(1, motion.fit_path(1, True))
        np.testing.assert_allclose(spot['scales'], [.32, .11, .08])
        projected, = project_splats(make_scene(spot), translated_camera(0, 80), 97, 61)
        np.testing.assert_allclose(projected['mean'], [46, 30])
        self.assertGreater(projected['covariance'][0, 1], 0)

    def test_rigidity_prior_helps_noisy_rigid_part_but_biases_exact_hinge(self):
        result = motion.persistent_pair()
        np.testing.assert_allclose(result['fitted'], [.08, .76])
        self.assertAlmostEqual(result['tracking_error'], .4472135955/5)
        self.assertEqual(motion.persistent_pair(noise=0, bend=40, strength=0)['tracking_error'], 0)
        hinged = motion.persistent_pair(noise=0, bend=40)
        self.assertGreater(hinged['tracking_error'], .4)
