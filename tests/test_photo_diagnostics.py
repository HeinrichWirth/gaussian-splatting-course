import json
import unittest

import numpy as np

import photo_diagnostics as photo


class PhotoDiagnosticsTests(unittest.TestCase):
    def test_browser_track_contains_measured_features(self):
        source = photo.feature_track(148)
        exported = photo.ROOT / 'site/dist/assets/reconstruction/track-148.json'
        self.assertEqual(json.loads(exported.read_text('utf8')), source)
        frames, _ = photo.photo_capture(max_side=80)
        a, b, c = [next(f for f in frames if f['name'] == name)
                   for name in ('view-03.jpg', 'view-08.jpg', 'view-07.jpg')]
        ua, ub, uc = [photo.measured_pixel(source, f) for f in (a, b, c)]
        for dx, expected in [(0, .40), (3, 2.01)]:
            point, _ = photo.intersect_observations(a, ua, b, ub + [dx, 0])
            self.assertAlmostEqual(np.linalg.norm(photo.reproject(point, c)-uc),
                                   expected, delta=.01)
        with self.assertRaises(ValueError):
            photo.intersect_observations(a, ua, a, ua)

    def test_attribution_and_region_match_rendered_photo(self):
        scene, frame = photo.reference_photo()
        target = frame['rgb']
        h, w = target.shape[:2]
        prediction = photo.render_scene(scene, frame['camera'], w, h)
        for u, v in ((40, 30), (0, 0)):
            rows, background, rgb = photo.pixel_ledger(scene, frame, u, v)
            self.assertAlmostEqual(sum(row[-1] for row in rows) + background, 1.)
            np.testing.assert_allclose(rgb, prediction[v, u], atol=1e-8)
        mask = np.zeros((h, w), dtype=bool)
        mask[15:45, 20:60] = True
        count, mse, share = photo.region_error(target, prediction, mask)
        self.assertEqual(count, 1200)
        self.assertAlmostEqual(mse, .01193155)
        self.assertAlmostEqual(share, mse*count/(h*w*np.mean((prediction-target)**2)))
