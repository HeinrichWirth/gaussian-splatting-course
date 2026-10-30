"""Numerical checks for the image-loss and descent examples."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import loss_demos as ld


class LossTests(unittest.TestCase):
    def test_four_pixels_and_rgb_denominator(self):
        result = ld.image_errors([[.2,.8],[.1,.9]], [[0,1],[.5,.5]])
        self.assertAlmostEqual(result["bias"], 0)
        self.assertAlmostEqual(result["mae"], .3)
        self.assertAlmostEqual(result["mse"], .1)
        rgb = ld.image_errors([[[.4,.2,.8]]], [[[.2,.5,.8]]])
        self.assertEqual(rgb["count"], 3)
        self.assertAlmostEqual(rgb["mae"], 1/6)
        self.assertAlmostEqual(rgb["mse"], .13/3)

    def test_scale_and_duplication(self):
        target, prediction = np.array([0,.5,1]), np.array([.1,.8,.7])
        original = ld.image_errors(prediction, target)
        duplicated = ld.image_errors(np.tile(prediction,5), np.tile(target,5))
        scaled = ld.image_errors(255*prediction,255*target)
        for metric in ("mae", "mse"):
            self.assertAlmostEqual(original[metric], duplicated[metric])
        self.assertAlmostEqual(scaled["mae"], 255*original["mae"])
        self.assertAlmostEqual(scaled["mse"], 255**2*original["mse"])

    def test_equal_mse_different_spatial_errors(self):
        concentrated = np.zeros((8,8)); concentrated[3,3] = .8
        target = np.zeros((8,8))
        self.assertAlmostEqual(ld.image_errors(np.full((8,8),.1),target)["mse"], .01)
        self.assertAlmostEqual(ld.image_errors(concentrated,target)["mse"], .01)

    def test_finite_difference_against_independent_analytic_derivative(self):
        x = np.linspace(-4,4,81)
        for center in (-2.,-1.,.3,1.,2.):
            prediction = np.exp(-.5*(x-center)**2)
            target = np.exp(-.5*(x-1)**2)
            expected = np.mean(2*(prediction-target)*prediction*(x-center))
            self.assertAlmostEqual(ld.central_difference(ld.center_loss,center,1e-4), expected, places=8)

    def test_multi_parameter_gradient(self):
        x = np.linspace(-4,4,81)
        mu, gain = -.7,.6
        shape = np.exp(-.5*(x-mu)**2)
        residual = gain*shape - ld.bell(1)
        expected = [np.mean(2*residual*gain*shape*(x-mu)), np.mean(2*residual*shape)]
        actual = ld.numerical_gradient(lambda p: ld.center_loss(*p), [mu,gain], 1e-4)
        np.testing.assert_allclose(actual,expected,atol=1e-9)

    def test_descent_and_overshoot(self):
        quadratic = lambda theta: (theta-3)**2
        np.testing.assert_allclose(ld.descent_steps(quadratic,1,.25,1), [[1,4],[2,1]], atol=1e-12)
        good = ld.descent_steps(ld.center_loss,-1,2,20)
        self.assertLess(good[-1,1],1e-9)
        self.assertTrue(np.all(np.diff(good[:,1]) < 0))
        bad = ld.descent_steps(quadratic,1,1.1,4)
        self.assertGreater(bad[-1,1],bad[0,1])

    def test_cropped_image_can_reward_disappearing_blob(self):
        self.assertLess(ld.central_difference(ld.center_loss,4),0)
        self.assertLess(ld.center_loss(4.1),ld.center_loss(4))
        self.assertGreater(ld.center_loss(4.1),ld.center_loss(1))

    def test_invalid_values_fail_loudly(self):
        for a,b in (([],[]), ([1,2],[1]), ([np.nan],[0])):
            with self.assertRaises(ValueError): ld.image_errors(a,b)
        for h in (0,-1,np.nan):
            with self.assertRaises(ValueError): ld.central_difference(lambda x:x*x,1,h)
        with self.assertRaises(ValueError): ld.gaussian_image(sigma=0)
        with self.assertRaises(ValueError): ld.descent_steps(lambda x:x*x,1,0,3)


if __name__ == "__main__":
    unittest.main()
