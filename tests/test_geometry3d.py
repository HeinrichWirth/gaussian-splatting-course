import unittest

import numpy as np

from geometry3d import (
    axis_angle_quaternion, covariance_3d, project_point, projected_covariance,
    projection_jacobian, quaternion_matrix, world_to_camera,
)


class GeometryTests(unittest.TestCase):
    def test_world_camera_uses_inverse_rotation_and_relative_position(self):
        camera = np.array([1., -2., 3.])
        basis = quaternion_matrix(axis_angle_quaternion([0, 1, 0], 90))
        result = world_to_camera(camera + [4, 1, -2], camera, basis)
        np.testing.assert_allclose(result, [2, 1, 4], atol=1e-12)
        np.testing.assert_allclose(camera + basis @ result, [5, -1, 1])
        with self.assertRaises(ValueError):
            world_to_camera([1, 2, 3], camera, np.diag([1, 1, -1]))

    def test_projected_reference_and_safe_clipping(self):
        np.testing.assert_allclose(project_point([1, .5, 4], (80, 100), (100, 60)), [120, 72.5])
        for z in [-2., 0., .2]:
            with self.assertRaises(ValueError):
                project_point([1, .5, z], 80, near=.2)

    def test_jacobian_is_local_derivative_of_actual_perspective(self):
        p = np.array([1.1, -.3, 2.7])
        focal = (80., 115.)
        eps = 1e-5
        side = lambda v: project_point(v, focal, (31, 19))
        slopes = [(side(p+eps*axis)-side(p-eps*axis))/(2*eps) for axis in np.eye(3)]
        np.testing.assert_allclose(projection_jacobian(p, focal), np.asarray(slopes).T, atol=1e-8)

    def test_linearized_ellipse_preserves_rotated_local_second_moment(self):
        q = axis_angle_quaternion([1, 2, -3], 64)
        scales = np.array([1., .75, .4])
        p = np.array([1.1, .3, 4.])
        R = quaternion_matrix(q)
        # Six symmetric points with these empirical second moments.
        axes = np.vstack([np.eye(3), -np.eye(3)]) * scales * np.sqrt(3)
        samples = axes @ R.T
        on_screen = samples @ projection_jacobian(p, 80).T
        np.testing.assert_allclose(on_screen.T @ on_screen / 6,
                                   projected_covariance(p, covariance_3d(scales, -3*q), 80))
        np.testing.assert_allclose(projected_covariance([1, 0, 4], np.diag([.25,.09,1]), 80),
                                   np.diag([125, 36]))

    def test_singular_shapes_and_axes(self):
        with self.assertRaises(ValueError):
            covariance_3d([.5, 0, 1], [1, 0, 0, 0])
        with self.assertRaises(ValueError):
            axis_angle_quaternion([0, 0, 0], 30)
        with self.assertRaises(ValueError):
            quaternion_matrix([0, 0, 0, 0])
