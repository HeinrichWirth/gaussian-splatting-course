"""Exercise real notebook controls and the figures their callbacks display."""

from contextlib import contextmanager
import copy
import gc
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from covariance_demos import covariance_lab
from gaussian_math import make_layer, scene_json
from mixing_demos import gaussian_editor, preset_scene
from plane_demos import plane_lab


class DisplayCapture:
    """Keep only the current figure, so repeated redraws do not retain images."""

    def __init__(self):
        self.figure = None
        self.html = ""
        self.count = 0
        self.open_figures_at_display = []

    def __call__(self, value, *args, **kwargs):
        if isinstance(value, Figure):
            self.figure = value
            self.count += 1
            self.open_figures_at_display.append(tuple(plt.get_fignums()))
        elif hasattr(value, "data") and isinstance(value.data, str):
            self.html = value.data


def close_widgets(widget, seen=None):
    seen = set() if seen is None else seen
    if id(widget) in seen:
        return
    seen.add(id(widget))
    for child in getattr(widget, "children", ()):
        close_widgets(child, seen)
    for key in ("layout", "style"):
        child = getattr(widget, key, None)
        if child is not None:
            close_widgets(child, seen)
    widget.close()


@contextmanager
def live_lab(factory, language="en"):
    capture = DisplayCapture()
    # Intercept only notebook output. The controls, observers, rendering, and
    # numerical calculations remain real, including all matplotlib figures.
    with patch("IPython.display.display", new=capture), patch(
            "IPython.display.clear_output", new=lambda **kwargs: None):
        lab = factory(language=language)
        try:
            yield lab, lab.course_controls, capture
        finally:
            close_widgets(lab)
            capture.figure = None
            gc.collect()


class PlaneWidgetTests(unittest.TestCase):
    def assert_closed_redraws(self, capture):
        self.assertGreater(capture.count, 0)
        self.assertTrue(all(not figures for figures in capture.open_figures_at_display))
        self.assertEqual(plt.get_fignums(), [])

    def assert_values(self, controls, names, expected):
        self.assertEqual(tuple(controls[name].value for name in names), expected)

    def test_plane_presets_slices_and_reset_render_current_values(self):
        names = ("mux", "muy", "sx", "sy", "x", "y")
        presets = {
            "circle": (0, 0, 1, 1, 1, 0),
            "vertical": (0, 0, 1, 2, 1, 2),
            "one_width": (0, 0, 2, 1, 2, 0),
            "center": (0, 0, 2, 1, 0, 0),
        }
        for language in ("ru", "en"):
            with self.subTest(language=language), live_lab(plane_lab, language) as (lab, c, shown):
                self.assertEqual(set(c), set(names) | {"preset", "view", "reset"})
                self.assertFalse(c["sx"].continuous_update)
                for name, expected in presets.items():
                    before = shown.count
                    c["preset"].value = name
                    self.assert_values(c, names, expected)
                    self.assertEqual(shown.count, before + 1)
                    image = shown.figure.axes[0].images[0]
                    self.assertEqual(image.get_clim(), (0, 1))
                    self.assertEqual(shown.figure.axes[0].get_xlim(), (-5, 5))
                    self.assertEqual(shown.figure.axes[0].get_ylim(), (-5, 5))

                c["view"].value = "slices"
                c["y"].value = 1
                cut = shown.figure.axes[1].lines[1]
                # Cutting at y=one vertical scale lowers the entire x curve.
                self.assertAlmostEqual(float(np.max(cut.get_ydata())), math.exp(-.5))
                c["mux"].value = 1
                cut = shown.figure.axes[1].lines[1]
                self.assertAlmostEqual(float(cut.get_xdata()[np.argmax(cut.get_ydata())]), 1)
                before = shown.count
                lab.course_redraw()
                self.assertEqual(shown.count, before + 1)

                c["reset"].click()
                self.assert_values(c, names, (0, 0, 2, 1, 2, 1))
                self.assertEqual(c["preset"].value, "default")
                self.assertEqual(c["view"].value, "views")
                self.assert_closed_redraws(shown)

    def test_covariance_controls_rotate_actual_image_and_reset(self):
        names = ("s1", "s2", "angle", "x", "y")
        presets = {
            "axis": (2, 1, 0, 1, 1),
            "opposite": (2, 1, -45, 1, 1),
            "circle": (1.5, 1.5, 45, 1, 1),
            "center": (2, 1, 45, 0, 0),
        }
        for language in ("ru", "en"):
            with self.subTest(language=language), live_lab(covariance_lab, language) as (lab, c, shown):
                self.assertEqual(set(c), set(names) | {"preset", "reset"})
                # The displayed grid has 241 samples from -4 to +4. (1,1) is
                # sample [150,150], on the long axis at +45 degrees.
                self.assertAlmostEqual(float(shown.figure.axes[0].images[0].get_array()[150, 150]), math.exp(-.25))
                for name, expected in presets.items():
                    before = shown.count
                    c["preset"].value = name
                    self.assert_values(c, names, expected)
                    self.assertEqual(shown.count, before + 1)
                    if name == "opposite":
                        self.assertAlmostEqual(float(shown.figure.axes[0].images[0].get_array()[150, 150]), math.exp(-1))
                c["angle"].value = 90
                # After the turn, (0,2) lies one long-axis scale away.
                self.assertAlmostEqual(float(shown.figure.axes[0].images[0].get_array()[180, 120]), math.exp(-.5))
                before = shown.count
                lab.course_redraw()
                self.assertEqual(shown.count, before + 1)
                c["reset"].click()
                self.assert_values(c, names, (2, 1, 45, 1, 1))
                self.assertEqual(c["preset"].value, "default")
                self.assert_closed_redraws(shown)

    def test_every_editor_preset_roundtrips_through_export_and_import_buttons(self):
        for language in ("ru", "en"):
            with self.subTest(language=language), live_lab(gaussian_editor, language) as (lab, c, shown):
                for preset in ("two_spots", "flower", "blank"):
                    with self.subTest(preset=preset):
                        c["preset"].value = preset
                        c["load_preset"].click()
                        expected = copy.deepcopy(lab.course_state)
                        c["export"].click()
                        saved = c["json"].value
                        self.assertEqual(json.loads(saved), json.loads(lab.course_export()))
                        c["preset"].value = "blank"
                        c["load_preset"].click()
                        c["json"].value = saved
                        before = shown.count
                        c["import"].click()
                        self.assertEqual(lab.course_state, expected)
                        self.assertEqual(shown.count, before + 1)

                # Every built-in layer must fit the controls exactly. Previously
                # the flower's narrow stem was clamped during selection, then
                # silently widened by an unrelated geometry edit.
                c["preset"].value = "flower"
                c["load_preset"].click()
                c["selected"].value = len(lab.course_state["layers"]) - 1
                original = copy.deepcopy(lab.course_state["layers"][-1])
                self.assertEqual(c["sx"].value, original["scales"][0])
                c["y"].value = -1
                expected = copy.deepcopy(original)
                expected["center"][1] = -1
                self.assertEqual(lab.course_state["layers"][-1], expected)
                self.assert_closed_redraws(shown)

    def test_editor_empty_canvas_add_remove_and_full_reset(self):
        for language in ("ru", "en"):
            with self.subTest(language=language), live_lab(gaussian_editor, language) as (lab, c, shown):
                c["preset"].value = "blank"
                c["load_preset"].click()
                self.assertEqual(lab.course_state["layers"], [])
                self.assertIsNone(c["selected"].value)
                for name in ("selected", "x", "y", "sx", "sy", "angle", "color", "opacity", "remove", "front", "back"):
                    self.assertTrue(c[name].disabled, name)
                np.testing.assert_array_equal(shown.figure.axes[0].images[0].get_array(), np.ones((161, 161, 3)))
                c["add"].click()
                self.assertEqual(len(lab.course_state["layers"]), 1)
                self.assertEqual(c["selected"].value, 0)
                self.assertFalse(c["x"].disabled)
                c["x"].value = 1.25
                self.assertEqual(lab.course_state["layers"][0]["center"][0], 1.25)
                c["remove"].click()
                self.assertEqual(lab.course_state["layers"], [])
                self.assertIsNone(c["selected"].value)
                c["probe_x"].value = 2
                c["probe_y"].value = -1
                c["outline"].value = False
                c["background"].value = "#123456"
                c["json"].value = "unsaved text"
                c["reset"].click()
                layers, background = preset_scene(language=language)
                self.assertEqual(lab.course_state, {"layers": layers, "background": background})
                self.assert_values(c, ("probe_x", "probe_y", "outline", "json", "preset"), (0, 0, True, "", "two_spots"))
                before = shown.count
                lab.course_redraw()
                self.assertEqual(shown.count, before + 1)
                self.assertIn("<table", shown.html)
                self.assert_closed_redraws(shown)

    def test_editor_reordering_keeps_selected_layer_and_updates_pixels(self):
        with live_lab(gaussian_editor) as (lab, c, shown):
            original = copy.deepcopy(lab.course_state["layers"])
            np.testing.assert_allclose(shown.figure.axes[0].images[0].get_array()[80, 80], [.75, .25, .5])
            c["back"].click()
            self.assertEqual(lab.course_state["layers"], original[::-1])
            self.assertEqual(c["selected"].value, 1)
            self.assertEqual(c["color"].value, "#ff0000")
            np.testing.assert_allclose(shown.figure.axes[0].images[0].get_array()[80, 80], [.5, .25, .75])
            c["x"].value = 1.25
            c["sy"].value = .65
            c["angle"].value = 23
            c["opacity"].value = .35
            edited = copy.deepcopy(lab.course_state["layers"][1])
            c["front"].click()
            self.assertEqual(c["selected"].value, 0)
            self.assertEqual(lab.course_state["layers"], [edited, original[1]])
            self.assertTrue(c["front"].disabled)
            c["selected"].value = 1
            self.assert_values(c, ("x", "sy", "angle", "opacity"), (0, 1, 0, .5))
            self.assertEqual(c["color"].value, "#0000ff")
            c["selected"].value = 0
            self.assert_values(c, ("x", "sy", "angle", "opacity"), (1.25, .65, 23, .35))
            self.assertEqual(lab.course_state["layers"][0], edited)
            self.assert_closed_redraws(shown)

    def test_editor_invalid_import_is_atomic_for_state_selection_and_render(self):
        with live_lab(gaussian_editor) as (lab, c, shown):
            c["selected"].value = 1
            c["x"].value = 2
            baseline = copy.deepcopy(lab.course_state)
            controls_before = tuple(c[key].value for key in ("selected", "x", "sx", "color", "background"))
            payload = json.loads(lab.course_export())
            invalid = ["not json"]
            wrong_version = copy.deepcopy(payload)
            wrong_version["version"] = 99
            invalid.append(json.dumps(wrong_version))
            # A valid first layer must not be installed before a later layer
            # fails either schema validation or the editor's slider bounds.
            for key, value in (("center", [4.1, 0]), ("scales", [.01, 1]),
                               ("angle_deg", 181), ("color", [1.1, 0, 0])):
                bad = copy.deepcopy(payload)
                bad["layers"][0]["center"] = [-3, -3]
                bad["layers"][1][key] = value
                invalid.append(json.dumps(bad))
            too_many = copy.deepcopy(payload)
            too_many["layers"] = [copy.deepcopy(payload["layers"][0]) for _ in range(129)]
            invalid.append(json.dumps(too_many))
            for text in invalid:
                with self.subTest(payload=text[:60]):
                    before = shown.count
                    with self.assertRaises(ValueError):
                        lab.course_import(text)
                    c["json"].value = text
                    c["import"].click()
                    self.assertEqual(lab.course_state, baseline)
                    self.assertEqual(tuple(c[key].value for key in ("selected", "x", "sx", "color", "background")), controls_before)
                    self.assertEqual(shown.count, before)
            self.assert_closed_redraws(shown)

    def test_editor_accepts_a_full_trained_scene(self):
        layers = [make_layer(f"Trained {i}", center=((i % 16) / 2 - 4, (i // 16) - 4),
                             scales=(.2, .3), opacity=.2) for i in range(128)]
        with live_lab(gaussian_editor) as (lab, c, shown):
            lab.course_import(scene_json(layers))
            self.assertEqual(len(lab.course_state["layers"]), 128)
            self.assertTrue(c["add"].disabled)
            c["selected"].value = 127
            self.assertEqual(lab.course_state["layers"][127]["name"], "Trained 127")
            self.assertEqual(len(json.loads(lab.course_export())["layers"]), 128)
            self.assert_closed_redraws(shown)

    def test_imported_rgb_precision_survives_selection_and_geometry_edits(self):
        color = [.123456789, .234567891, .876543219]
        background = [.234567891, .345678912, .456789123]
        layers = [make_layer("Precise", color=color, scales=(.731, 1.234), opacity=.6789),
                  make_layer("Other", color=(.4, .5, .6))]
        with live_lab(gaussian_editor) as (lab, c, shown):
            c["json"].value = scene_json(layers, background)
            c["import"].click()
            c["x"].value = .35
            c["angle"].value = 17
            c["selected"].value = 1
            c["selected"].value = 0
            self.assertEqual(lab.course_state["layers"][0]["color"], color)
            self.assertEqual(lab.course_state["layers"][0]["scales"], [.731, 1.234])
            self.assertEqual(lab.course_state["layers"][0]["opacity"], .6789)
            self.assertEqual(lab.course_state["background"], background)
            c["export"].click()
            exported = json.loads(c["json"].value)
            self.assertEqual(exported["layers"][0]["color"], color)
            self.assertEqual(exported["background"], background)
            # Only an intentional color-picker edit should quantize to 8 bits.
            c["color"].value = "#123456"
            self.assertEqual(lab.course_state["layers"][0]["color"], [18/255, 52/255, 86/255])
            self.assert_closed_redraws(shown)


if __name__ == "__main__":
    unittest.main()
