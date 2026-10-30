"""Numerical checks against independent calculations and the lesson 5 renderer."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gaussian_math import load_scene_json, render_layers
from training_math import (autograd_check, chain_example, fit_image, fit_one_gaussian,
                           initialize, load_image, make_target, parameters_scene, projected_step,
                           render_torch, small_cpu_work)


class ChainArithmetic(unittest.TestCase):
    def test_every_local_factor_matches_a_small_perturbation(self):
        c = chain_example()
        functions = [lambda x: x * x, np.exp, lambda x: -x / 2,
                     lambda x: x * x, lambda x: x, lambda x: 1 - x]
        inputs = [c["residual"], c["t"], c["q"], c["u"], c["d"], 0.0]
        for function, point, expected in zip(functions, inputs, c["local"]):
            numerical = (function(point + 1e-6) - function(point - 1e-6)) / 2e-6
            self.assertAlmostEqual(numerical, expected, places=8)


@unittest.skipIf(importlib.util.find_spec("torch") is None, "requires requirements-training.txt")
class AutogradAndImageFit(unittest.TestCase):
    def test_chain_torch_and_difference_agree(self):
        for center in [-1.0, 0.0, .7, 1.5]:
            values = autograd_check(center)
            self.assertAlmostEqual(values["chain"], values["autograd"], places=12)
            self.assertAlmostEqual(values["chain"], values["finite_difference"], places=8)

    def test_single_gaussian_reaches_the_target(self):
        result = fit_one_gaussian()
        self.assertAlmostEqual(result["history"][-1]["center"], 1.5, places=5)
        self.assertLess(result["history"][-1]["loss"], 1e-10)
        self.assertEqual(len(result["history"]), 81)

    def test_torch_renderer_matches_saved_scene_renderer(self):
        parameters = initialize(make_target(size=16), count=16, placement="random", seed=42)
        layers, background = load_scene_json(parameters_scene(parameters))
        expected = render_layers(layers, background, size=16)
        actual = render_torch(parameters, 16).detach().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_compositing_gradients_by_finite_difference(self):
        import torch
        target = make_target(size=8)
        parameters = initialize(target, count=3, placement="random", seed=19, scale=1.2)
        with small_cpu_work():
            loss = (render_torch(parameters, 8) - torch.tensor(target)).square().mean()
            loss.backward()
            for name, parameter in parameters.items():
                for flat in range(parameter.numel()):
                    index = np.unravel_index(flat, parameter.shape)
                    expected = parameter.grad[index].item()
                    old = parameter[index].item()
                    values = []
                    with torch.no_grad():
                        for delta in [-1e-5, 1e-5]:
                            parameter[index] = old + delta
                            values.append(((render_torch(parameters, 8) - torch.tensor(target)).square().mean()).item())
                        parameter[index] = old
                    self.assertAlmostEqual((values[1] - values[0]) / 2e-5, expected, places=8,
                                           msg=f"{name}{index}")

    def test_image_fit_lowers_loss_and_roundtrips(self):
        result = fit_image(make_target(size=16), count=16, steps=60, scale=.7)
        self.assertLess(result["history"][-1], result["history"][0] * .65)
        layers, background = load_scene_json(result["scene"])
        np.testing.assert_allclose(render_layers(layers, background, size=16), result["image"], atol=1e-12)
        self.assertEqual(len(layers), 16)

    def test_zero_rate_keeps_image_and_seed_repeats(self):
        a = fit_image(make_target(size=8), count=4, steps=2, placement="random", rate_multiplier=0)
        b = fit_image(make_target(size=8), count=4, steps=2, placement="random", rate_multiplier=0)
        self.assertEqual(a["scene"], b["scene"])
        np.testing.assert_array_equal(a["snapshots"][0], a["image"])

    def test_projection_and_gradient_reset(self):
        import torch
        parameters = initialize(make_target(size=8), count=1)
        for value in parameters.values():
            value.grad = torch.full_like(value, 100.0)
        projected_step(parameters)
        self.assertTrue(all(value.grad is None for value in parameters.values()))
        self.assertTrue((parameters["scales"] == .2).all())
        self.assertTrue((parameters["colors"] == 0).all())
        json.loads(parameters_scene(parameters))

    def test_fractional_steps_and_dimensions_are_rejected(self):
        for value in [1.5, True, float("inf"), float("nan")]:
            with self.assertRaises(ValueError):
                fit_one_gaussian(steps=value)
            with self.assertRaises(ValueError):
                fit_image(steps=value)
        for value in [32.5, True, float("inf")]:
            with self.assertRaises(ValueError):
                make_target(size=value)
            with self.assertRaises(ValueError):
                load_image("not-read.png", size=value)

    def test_invalid_update_does_not_partially_change_parameters(self):
        import torch
        parameters = initialize(make_target(size=8), count=2)
        before = {name: value.detach().clone() for name, value in parameters.items()}
        for value in parameters.values():
            value.grad = torch.ones_like(value)
        for rates in [{"centers": 4}, dict(centers=4, scales=4, colors=4, opacity=float("nan")),
                      dict(centers=4, scales=4, colors=4, opacity=-1)]:
            with self.assertRaises(ValueError):
                projected_step(parameters, rates)
            for name, value in parameters.items():
                self.assertTrue(torch.equal(before[name], value))

    def test_local_rgba_image_preserves_aspect_and_vertical_orientation(self):
        from PIL import Image
        source = np.zeros((12, 6, 4), dtype=np.uint8)
        source[:6] = [255, 0, 0, 255]
        source[6:] = [0, 0, 255, 255]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "portrait.png"
            Image.fromarray(source).save(path)
            loaded = load_image(path, size=8)
        self.assertEqual(loaded.shape, (8, 8, 3))
        np.testing.assert_array_equal(loaded[:, 0], np.ones((8, 3)))
        self.assertGreater(loaded[-1, 3, 0], .95)  # top is red, y positive
        self.assertGreater(loaded[0, 3, 2], .95)   # bottom is blue, y negative

    def test_opaque_overlap_has_finite_gradients(self):
        import torch
        parameters = initialize(make_target(size=8), count=2)
        with torch.no_grad():
            parameters["centers"].zero_()
            parameters["opacity"].fill_(1)
        with small_cpu_work():
            render_torch(parameters, 9).sum().backward()
        self.assertTrue(all(torch.isfinite(value.grad).all() for value in parameters.values()))


if __name__ == "__main__":
    unittest.main()
