"""Check the crop, pose convention and split of the bundled photographic exercise."""
import unittest
from pathlib import Path

import numpy as np

from reconstruction import load_colmap

ROOT = Path(__file__).resolve().parents[1]


class PhotoCalibrationTests(unittest.TestCase):
    def test_sparse_tracks_land_on_their_annotated_training_pixels(self):
        data = ROOT / 'data' / 'photographic-example'
        model = data / 'sparse' / '0'
        frames, _ = load_colmap(model, data / 'images', max_side=160,
                                test_names=['view-04.jpg'])
        by_name = {frame['name']: frame for frame in frames}
        points = {}
        expected_errors = {}
        for row in (model / 'points3D.txt').read_text().splitlines():
            if row and not row.startswith('#'):
                fields = row.split()
                point_id = int(fields[0])
                points[point_id] = np.array(fields[1:4], dtype=float)
                expected_errors[point_id] = float(fields[7])
        tracks = iter(line for line in (model / 'images.txt').read_text().splitlines()
                      if not line.startswith('#'))
        errors = {point_id: [] for point_id in points}
        for pose in tracks:
            frame = by_name[pose.split(maxsplit=9)[-1]]
            measurements = np.fromstring(next(tracks), sep=' ').reshape(-1, 3)
            if frame['split'] == 'test':
                self.assertEqual(len(measurements), 0)
                continue
            camera = frame['camera']
            for u, v, point_id in measurements:
                p = (points[int(point_id)] - camera['center']) @ camera['rotation']
                pixel = camera['principal'] + np.array(camera['focal']) * p[:2] / p[2]
                # COLMAP measures from the corner; renderer coordinates start at pixel center.
                errors[int(point_id)].append(np.linalg.norm(pixel - [u-.5, v-.5]))
        for point_id, residuals in errors.items():
            self.assertGreaterEqual(len(residuals), 2)
            self.assertAlmostEqual(np.mean(residuals), expected_errors[point_id], places=6)
