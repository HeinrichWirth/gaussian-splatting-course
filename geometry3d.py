"""Camera coordinates and local Gaussian footprints. Rotations use scalar-first quaternions."""
import numpy as np


def _array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _unit(value, length, name):
    vector = _array(value, (length,), name)
    largest = np.max(np.abs(vector))
    if largest == 0:
        raise ValueError(f"{name} must have nonzero length")
    vector = vector / largest
    return vector / np.linalg.norm(vector)


def axis_angle_quaternion(axis, angle_deg):
    """Encode an active right-handed rotation in (w, x, y, z) order."""
    unit = _unit(axis, 3, "axis")
    if not np.isfinite(angle_deg):
        raise ValueError("angle must be finite")
    half = np.deg2rad(angle_deg) / 2
    return np.r_[np.cos(half), unit * np.sin(half)]


def quaternion_matrix(quaternion):
    w, x, y, z = _unit(quaternion, 4, "quaternion")
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def world_to_camera(point, camera_center, camera_to_world):
    """Camera-to-world columns are the right, down and forward world vectors."""
    p = _array(point, (3,), "point")
    center = _array(camera_center, (3,), "camera center")
    axes = _array(camera_to_world, (3, 3), "camera basis")
    if not (np.allclose(axes.T @ axes, np.eye(3)) and np.isclose(np.linalg.det(axes), 1)):
        raise ValueError("camera basis must be a proper rotation")
    return axes.T @ (p - center)


def _focal_pair(focal):
    pair = np.broadcast_to(np.asarray(focal, dtype=float), (2,))
    if not np.all(np.isfinite(pair) & (pair > 0)):
        raise ValueError("focal lengths must be finite and positive")
    return pair


def project_point(camera_xyz, focal, principal=(0, 0), near=0.05):
    p = _array(camera_xyz, (3,), "camera point")
    if not np.isfinite(near) or near < 0 or p[2] <= near:
        raise ValueError("point must lie beyond the nonnegative near depth")
    return _array(principal, (2,), "principal point") + _focal_pair(focal) * p[:2] / p[2]


def projection_jacobian(camera_xyz, focal):
    x, y, z = _array(camera_xyz, (3,), "camera point")
    if z <= 0:
        raise ValueError("depth must be positive")
    fx, fy = _focal_pair(focal)
    return np.array([[fx/z, 0, -fx*x/z**2], [0, fy/z, -fy*y/z**2]])


def covariance_3d(scales, quaternion):
    scales = _array(scales, (3,), "axis scales")
    if np.any(scales <= 0):
        raise ValueError("axis scales must be positive")
    basis = quaternion_matrix(quaternion) * scales
    return basis @ basis.T


def projected_covariance(camera_xyz, camera_covariance, focal):
    cov = _array(camera_covariance, (3, 3), "camera covariance")
    if not np.allclose(cov, cov.T) or np.linalg.eigvalsh(cov)[0] <= 0:
        raise ValueError("covariance must be symmetric and positive definite")
    j = projection_jacobian(camera_xyz, focal)
    return j @ cov @ j.T
