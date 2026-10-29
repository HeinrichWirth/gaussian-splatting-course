"""Independent numerical examples used to check the first block's experiments."""

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gaussian_math as gm


class PlaneMathTests(unittest.TestCase):
    def test_two_coordinates_and_equal_value_ellipse(self):
        self.assertAlmostEqual(float(gm.gaussian2d(0, 0)), 1)
        self.assertAlmostEqual(float(gm.gaussian2d(2, 0)), math.exp(-.5))
        self.assertAlmostEqual(float(gm.gaussian2d(0, 1)), math.exp(-.5))
        self.assertAlmostEqual(float(gm.gaussian2d(2, 1)), math.exp(-1))
        self.assertAlmostEqual(float(gm.gaussian2d(-2, -1)), math.exp(-1))
        x = np.array([-2., 0., 2.])[:, None]
        y = np.array([-1., 0., 1.])[None, :]
        expected = np.exp(-x*x/8) * np.exp(-y*y/2)
        np.testing.assert_allclose(gm.gaussian2d(x, y), expected)

    def test_translation_and_width_do_not_change_peak(self):
        for center in ((0, 0), (1, -2), (-3, 2)):
            for scales in ((2, 1), (.3, 2.5), (1, 1)):
                self.assertAlmostEqual(float(gm.gaussian2d(*center, *center, *scales, 67)), 1)
        self.assertAlmostEqual(float(gm.gaussian2d(3, -1, mux=1, muy=-2)), math.exp(-1))

    def test_rotation_uses_local_axes(self):
        np.testing.assert_allclose(gm.rotation_matrix(90) @ [2, 0], [0, 2], atol=1e-14)
        np.testing.assert_allclose(gm.covariance_matrix(2, 1, 90), [[1, 0], [0, 4]], atol=1e-14)
        self.assertAlmostEqual(float(gm.gaussian2d(0, 2, angle_deg=90)), math.exp(-.5))
        for angle in (0, 30, 45, 90, 180, 317):
            self.assertAlmostEqual(float(gm.gaussian2d(1, 2, sx=2, sy=2, angle_deg=angle)), math.exp(-5/8))

    def test_covariance_and_inverse_worked_example(self):
        covariance = gm.covariance_matrix(2, 1, 45)
        inverse = gm.inverse_covariance(covariance)
        np.testing.assert_allclose(covariance, [[2.5, 1.5], [1.5, 2.5]])
        np.testing.assert_allclose(inverse, [[.625, -.375], [-.375, .625]])
        np.testing.assert_allclose(covariance @ inverse, np.eye(2), atol=1e-14)
        steps = gm.gaussian2d_steps(1, 1, angle_deg=45)
        self.assertAlmostEqual(steps["local_x"], math.sqrt(2))
        self.assertAlmostEqual(steps["local_y"], 0)
        self.assertAlmostEqual(steps["q"], .5)
        self.assertAlmostEqual(steps["weight"], math.exp(-.25))
        self.assertAlmostEqual(float(gm.gaussian2d(2, 0, angle_deg=45)), math.exp(-1.25))
        self.assertAlmostEqual(float(gm.gaussian2d(1, -1, angle_deg=-45)), math.exp(-.25))

    def test_quadratic_form_matches_direct_local_distance(self):
        points = np.array([[0., 0.], [1., 1.], [-2., 1.], [3., -2.]])
        center = np.array([.3, -.7])
        for angle in (0, 23, 45, 90, 165):
            inverse = np.linalg.inv(gm.covariance_matrix(2.3, .7, angle))
            d = points - center
            q = np.einsum("ni,ij,nj->n", d, inverse, d)
            expected = np.exp(-q/2)
            actual = gm.gaussian2d(points[:, 0], points[:, 1], *center, 2.3, .7, angle)
            np.testing.assert_allclose(actual, expected, rtol=1e-13)

    def test_sample_covariance_population_arithmetic(self):
        points = np.array([[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]])
        rotated = points @ gm.rotation_matrix(45).T
        centered = rotated - rotated.mean(axis=0)
        empirical = centered.T @ centered / 4
        np.testing.assert_allclose(empirical, [[2.5, 1.5], [1.5, 2.5]])

    def test_invalid_shapes_are_rejected(self):
        for matrix in ([[1, 1], [1, 1]], [[1, 2], [2, 1]],
                       [[-1, 0], [0, -1]], [[1, 1], [0, 1]],
                       [[1, 0], [0, float("nan")]], [1, 2]):
            with self.assertRaises(ValueError):
                gm.inverse_covariance(matrix)
        for kwargs in (dict(sx=0), dict(sy=-1), dict(mux=math.inf), dict(angle_deg=math.nan)):
            with self.assertRaises(ValueError):
                gm.gaussian2d(1, 1, **kwargs)


class CompositingTests(unittest.TestCase):
    def setUp(self):
        self.red = gm.make_layer("red", color=(1, 0, 0), opacity=.5)
        self.blue = gm.make_layer("blue", color=(0, 0, 1), opacity=.5)

    def test_order_and_background_by_hand(self):
        trace = gm.composite_at(0, 0, [self.red, self.blue])
        np.testing.assert_allclose(trace["color"], [.75, .25, .5])
        np.testing.assert_allclose(trace["layers"][0]["contribution"], [.5, 0, 0])
        np.testing.assert_allclose(trace["layers"][1]["contribution"], [0, 0, .25])
        np.testing.assert_allclose(trace["background_contribution"], [.25, .25, .25])
        self.assertEqual(trace["transmittance"], .25)
        swapped = gm.composite_at(0, 0, [self.blue, self.red])
        np.testing.assert_allclose(swapped["color"], [.5, .25, .75])

    def test_weight_and_opacity_are_different(self):
        trace = gm.composite_at(1, 0, [self.red])
        self.assertAlmostEqual(trace["layers"][0]["weight"], math.exp(-.5))
        self.assertAlmostEqual(trace["layers"][0]["alpha"], .5*math.exp(-.5))
        expected = .5*math.exp(-.5)*np.array([1, 0, 0]) + (1-.5*math.exp(-.5))
        np.testing.assert_allclose(trace["color"], expected)

    def test_transparent_and_opaque_front(self):
        transparent = gm.make_layer("invisible", opacity=0)
        opaque = gm.make_layer("opaque", color=(.2, .7, .4), opacity=1)
        np.testing.assert_allclose(gm.composite_at(0, 0, [transparent, self.blue])["color"],
                                   gm.composite_at(0, 0, [self.blue])["color"])
        trace = gm.composite_at(0, 0, [opaque, self.blue])
        np.testing.assert_allclose(trace["color"], [.2, .7, .4])
        self.assertEqual(trace["transmittance"], 0)
        np.testing.assert_array_equal(trace["layers"][1]["contribution"], 0)
        self.assertGreater(gm.composite_at(1, 0, [opaque])["transmittance"], 0)

    def test_grid_and_trace_match_reverse_painting(self):
        layers = [self.red, self.blue, gm.make_layer("green", center=(.5, -.25),
                  scales=(.6, 1.2), angle_deg=35, color=(0, 1, 0), opacity=.7)]
        background = np.array([.1, .2, .3])
        image = gm.render_layers(layers, background, size=17)
        for row, col in ((8, 8), (5, 11), (0, 16)):
            x, y = -4 + col*.5, -4 + row*.5
            painted = background.copy()
            for layer in reversed(layers):
                alpha = layer["opacity"] * gm.gaussian2d(x, y, *layer["center"],
                                                        *layer["scales"], layer["angle_deg"])
                painted = alpha*np.array(layer["color"]) + (1-alpha)*painted
            np.testing.assert_allclose(image[row, col], painted)
            np.testing.assert_allclose(gm.composite_at(x, y, layers, background)["color"], painted)
        np.testing.assert_allclose(gm.render_layers([], background, size=3),
                                   np.broadcast_to(background, (3, 3, 3)))

    def test_scene_roundtrip_and_validation(self):
        original = [self.red, self.blue]
        saved = gm.scene_json(original, (.2, .3, .4))
        layers, background = gm.load_scene_json(saved)
        self.assertEqual(layers, original)
        np.testing.assert_allclose(gm.render_layers(layers, background, size=9),
                                   gm.render_layers(original, (.2, .3, .4), size=9))
        layers[0]["center"][0] = 3
        self.assertEqual(original[0]["center"][0], 0)
        for bad in ("not json", "{}", '{"version":1}', gm.scene_json([]).replace('"version": 1', '"version": true')):
            with self.assertRaises(ValueError):
                gm.load_scene_json(bad)
        for kwargs in (dict(opacity=-.1), dict(opacity=1.1), dict(color=(1, 0, 2)),
                       dict(center=(0, math.nan)), dict(scales=(0, 1))):
            with self.assertRaises(ValueError):
                gm.make_layer(**kwargs)


if __name__ == "__main__":
    unittest.main()
