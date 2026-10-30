"""Numerical and publishing checks. Run: python -m unittest discover -s tests -v"""

import contextlib
import io
import math
from pathlib import Path
import re
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np

import course_demos as demos
from scripts.build_lessons import github_markdown

ROOT = Path(__file__).resolve().parents[1]


class GaussianTests(unittest.TestCase):
    def test_reference_values_and_symmetry(self):
        for x, expected in [(2, 1), (4, 0.6065306597126334),
                            (6, 0.1353352832366127), (8, 0.011108996538242306)]:
            self.assertAlmostEqual(demos.gaussian_steps(x, 2, 2)["weight"], expected, places=14)
            self.assertAlmostEqual(demos.gaussian_steps(4-x, 2, 2)["weight"], expected, places=14)
        steps = demos.gaussian_steps(0, 2, 2)
        self.assertEqual([steps[k] for k in ("offset", "standardized", "squared", "exponent")], [-2, -1, 1, -0.5])

    def test_scale_and_translation(self):
        self.assertAlmostEqual(demos.gaussian_steps(4, 2, 2)["weight"], demos.gaussian_steps(7, 3, 4)["weight"])
        self.assertLess(demos.gaussian_steps(4, 2, 1)["weight"], demos.gaussian_steps(4, 2, 2)["weight"])
        for sigma in (0.25, 1, 3):
            self.assertEqual(demos.gaussian_steps(2, 2, sigma)["weight"], 1)
        np.testing.assert_allclose(demos.gaussian_values([2, 4, 6, 8], 2, 2), [1, .6065306597126334, .1353352832366127, .011108996538242306])

    def test_invalid_inputs(self):
        for sigma in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                demos.gaussian_steps(4, 2, sigma)
        with self.assertRaises(ValueError):
            demos.gaussian_values([1, math.nan], 2, 1)

    def test_fixed_viewport_and_sample_independence(self):
        figures = [demos.gaussian_figure(mu=2, sigma=2, x=x, language="en") for x in (0, 4)]
        np.testing.assert_array_equal(figures[0].axes[0].lines[0].get_ydata(), figures[1].axes[0].lines[0].get_ydata())
        for fig in figures + [demos.gaussian_figure(mu=-3, sigma=.25, x=12)]:
            self.assertEqual(fig.axes[0].get_xlim(), (-12, 12))
            self.assertEqual(fig.axes[0].get_ylim(), (0, 1.05))


class SceneTests(unittest.TestCase):
    def test_scene_determinism_and_projection(self):
        one, two = demos.scene_data(), demos.scene_data()
        self.assertEqual(one["means"].shape, (192, 3))
        for key in one:
            np.testing.assert_array_equal(one[key], two[key])
        one["means"][0, 0] = 999
        self.assertNotEqual(demos.scene_data()["means"][0, 0], 999)
        for azimuth, elevation in [(0, 0), (-55, 24), (180, 80), (90, -10)]:
            means, covs, depth = demos.project_scene(two, azimuth, elevation)
            self.assertEqual(means.shape, (192, 2))
            self.assertTrue(np.all(np.linalg.eigvalsh(covs) > 0))
            self.assertTrue(np.all(np.isfinite(depth)))
        data = dict(means=np.array([[0., 0., .75]]), covariances=np.array([np.eye(3)*.25]),
                    colors=np.array([[1., 0., 0.]]), opacities=np.array([.5]))
        _, covs, _ = demos.project_scene(data, 43, 31)
        np.testing.assert_allclose(covs[0], np.eye(2)*.25, atol=1e-14)

    def test_alpha_compositing_at_pixel_center(self):
        background = np.array([244, 247, 251])/255
        data = dict(means=np.array([[-1., 0., .75], [1., 0., .75]]),
                    covariances=np.repeat((np.eye(3)*.04)[None], 2, axis=0),
                    colors=np.array([[1., 0., 0.], [0., 0., 1.]]), opacities=np.array([.5, .5]))
        image = demos.render_scene(data, azimuth=0, elevation=0, size=33)
        np.testing.assert_allclose(image[16,16], .5*data["colors"][1]+.25*data["colors"][0]+.25*background)
        reversed_data = {key: value[::-1].copy() for key, value in data.items()}
        np.testing.assert_allclose(image, demos.render_scene(reversed_data, azimuth=0, elevation=0, size=33))
        absent = demos.render_scene(data, azimuth=0, elevation=0, selected=0, opacity=0, isolate=True, size=33)
        np.testing.assert_allclose(absent, np.broadcast_to(background, absent.shape))
        opaque = demos.render_scene(data, azimuth=0, elevation=0, selected=1, opacity=1, isolate=True, size=33)
        np.testing.assert_allclose(opaque[16,16], data["colors"][1])
        self.assertTrue(np.any(opaque[0,0] != data["colors"][1]))


class WidgetTests(unittest.TestCase):
    def test_bilingual_controls_and_reset(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for language in ("ru", "en"):
                lab = demos.gaussian_lab(language)
                controls = lab.course_controls
                controls["preset"].value = "narrow"
                self.assertEqual(controls["sigma"].value, 1)
                controls["mu"].value = -3
                controls["x"].value = 12
                controls["reset"].click()
                self.assertEqual([controls[n].value for n in ("mu", "sigma", "x")], [2, 2, 4])
                lab.close()
                scene = demos.scene_lab(language)
                controls = scene.course_controls
                controls["azimuth"].value = 45
                controls["selected"].value = 128
                self.assertAlmostEqual(controls["opacity"].value, .88)
                controls["isolate"].value = True
                controls["opacity"].value = 0
                controls["mode"].value = "centers"
                self.assertEqual(controls["azimuth"].value, 45)
                controls["mode"].value = "contours"
                controls["mode"].value = "composite"
                controls["reset"].click()
                self.assertEqual(controls["azimuth"].value, -55)
                self.assertFalse(controls["isolate"].value)
                self.assertEqual(controls["selected"].value, 0)
                scene.close()
        self.assertEqual(plt.get_fignums(), [])


class MaterialTests(unittest.TestCase):
    def test_github_math_cannot_become_markdown_headings(self):
        # This exact layout broke on GitHub: bare equals signs became headings.
        source = "Before $x=1$.\n\n$$\nx^2\n=\nx\\cdot x\n$$\n\nAfter."
        expected = "Before $x=1$.\n\n```math\nx^2\n=\nx\\cdot x\n```\n\nAfter."
        self.assertEqual(github_markdown(source), expected)
        self.assertEqual(github_markdown("$$x^2=4$$"), "```math\nx^2=4\n```")

    def test_github_math_keeps_examples_and_details_intact(self):
        example = '```python\ntext = "$$"\n```\n\n'
        details = "<details>\n<summary>Answer</summary>\n\n$$\nx=1\n$$\n\n</details>"
        expected = example + details.replace("$$\nx=1\n$$", "```math\nx=1\n```")
        self.assertEqual(github_markdown(example + details), expected)
        for source in ("$$\nx=1", "$$\n$$"):
            with self.assertRaises(ValueError):
                github_markdown(source)

    def test_github_math_rejects_unsupported_phantom(self):
        for source in (r"The root: $\sqrt{\phantom{9}}$.", "$$\n\\sqrt{\\phantom{9}}\n$$"):
            with self.assertRaisesRegex(ValueError, "phantom"):
                github_markdown(source)
        code = "```tex\n\\phantom{9}\n```"
        self.assertEqual(github_markdown(code), code)

    def test_github_math_rejects_unsupported_operatorname(self):
        for source in (r"The diagonal: $\operatorname{diag}(4,1)$.",
                       "$$\n\\operatorname{diag}(4,1)\n$$"):
            with self.assertRaisesRegex(ValueError, "operatorname"):
                github_markdown(source)
        code = "```tex\n\\operatorname{diag}(4,1)\n```"
        self.assertEqual(github_markdown(code), code)
        supported = r"The diagonal: $\mathrm{diag}(4,1)$."
        self.assertEqual(github_markdown(supported), supported)

    def test_every_display_formula_is_exported_as_fenced_math(self):
        for language in ("ru", "en"):
            for lesson in range(1, 10):
                nb = nbformat.read(ROOT/"notebooks"/language/f"lesson-{lesson:02}.ipynb", as_version=4)
                formulas = []
                for cell in nb.cells:
                    if cell.cell_type == "markdown":
                        formulas.extend(re.findall(r"(?ms)^\$\$\n(.*?)\n\$\$$", cell.source))
                suffix = "-en" if language == "en" else ""
                exported = (ROOT/"lessons"/language/f"lesson-{lesson:02}{suffix}.md").read_text(encoding="utf-8")
                rendered_formulas = re.findall(r"(?ms)^```math\n(.*?)\n```$", exported)
                self.assertEqual(rendered_formulas, formulas, f"{language}/lesson-{lesson:02}")
                self.assertNotRegex(exported, r"(?m)^\$\$$")

    def test_notebooks_executed_and_translations_structurally_match(self):
        for lesson in range(1, 10):
            notebooks = []
            for language in ("ru", "en"):
                path = ROOT / "notebooks" / language / f"lesson-{lesson:02}.ipynb"
                nb = nbformat.read(path, as_version=4)
                nbformat.validate(nb)
                images = 0
                for cell in nb.cells:
                    if cell.cell_type == "code":
                        self.assertIsNotNone(cell.execution_count, str(path))
                        for output in cell.outputs:
                            self.assertNotEqual(output.output_type, "error", str(path))
                            images += "image/png" in output.get("data", {})
                self.assertGreaterEqual(images, 2)
                notebooks.append(nb)
            self.assertEqual([c.cell_type for c in notebooks[0].cells], [c.cell_type for c in notebooks[1].cells])

    def test_local_links_and_exported_figures(self):
        files = [ROOT/"README.md", ROOT/"README_ru.md", *sorted((ROOT/"lessons").glob("*/*.md"))]
        for path in files:
            text = path.read_text(encoding="utf-8")
            for destination in re.findall(r"\]\(([^\s)]+)\)", text):
                if re.match(r"https?://|#|mailto:", destination):
                    continue
                target = destination.split("#", 1)[0]
                self.assertTrue((path.parent/target).exists(), f"{path}: {destination}")
            self.assertNotIn("application/vnd.jupyter.widget-view", text)
        for language in ("ru", "en"):
            for lesson in range(1, 10):
                self.assertGreaterEqual(len(list((ROOT/"assets"/f"lesson{lesson}"/language).glob("figure-*.png"))), 2)


if __name__ == "__main__":
    unittest.main()
