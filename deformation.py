"""One canonical ribbon, a deformation field and a motion fit from stereo tracks."""
from copy import deepcopy

import numpy as np

from splat_render import make_scene, splat

FORMAT = "gs-course-motion-v1"
TRAIN_TIMES = (.25, .75)
FOCAL = 64.
BASELINE = .8
PRINCIPAL = (32., 24.)


def ribbon():
    colors = [[.13,.28,.68], [.09,.51,.65], [.12,.65,.49], [.88,.65,.14], [.93,.34,.14]]
    return make_scene(*(splat([-.8+.4*i, 0., 4.], [.27,.15,.12], color, .93)
                        for i, color in enumerate(colors)))


def field_value(x_fraction, t, corner=1.):
    """Grid rows correspond to time, columns to x. Values are unitless."""
    if not 0 <= x_fraction <= 1 or not 0 <= t <= 1 or not np.isfinite(corner):
        raise ValueError("a query inside the unit square and a finite corner are required")
    grid = [[0., 0.], [0., corner]]
    early = (1-x_fraction)*grid[0][0] + x_fraction*grid[0][1]
    late = (1-x_fraction)*grid[1][0] + x_fraction*grid[1][1]
    return (1-t)*early + t*late


def field_scene(t, corner=1.):
    scene = ribbon()
    for spot in scene["splats"]:
        f = field_value((spot["center"][0]+1)/2, t, corner)
        spot["center"][1] -= .8*f
        angle = np.deg2rad(-35*f)
        spot["rotation"] = [float(np.cos(angle/2)), 0., 0., float(np.sin(angle/2))]
        spot["scales"][0] *= 1+.3*f
    return scene


def pulse(t):
    return 4*t*(1-t)


def validate_model(model):
    if not isinstance(model, dict) or model.get("format") != FORMAT:
        raise ValueError("unknown motion format")
    duration = model.get("duration")
    if not isinstance(duration, (float, int)) or isinstance(duration, bool) or not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive")
    scene = model.get("canonical")
    if not isinstance(scene, dict) or scene.get("format") != "gs-course-3d-v1":
        raise ValueError("missing canonical scene")
    spots = scene.get("splats")
    if not isinstance(spots, list) or not 1 <= len(spots) <= 128:
        raise ValueError("expected 1 to 128 splats")

    def numbers(values, shape):
        a = np.asarray(values)
        if a.shape != shape or a.dtype.kind not in "iuf" or not np.isfinite(a).all():
            raise ValueError("invalid numeric field")
        return a

    numbers(model.get("amplitudes"), (len(spots), 3))
    for item in spots:
        if not isinstance(item, dict):
            raise ValueError("invalid splat")
        numbers(item.get("center"), (3,))
        rgb = numbers(item.get("color"), (3,))
        scales = numbers(item.get("scales"), (3,))
        q = numbers(item.get("rotation"), (4,))
        opacity = numbers(item.get("opacity"), ())
        if (np.any(scales <= 0) or np.any((rgb < 0) | (rgb > 1)) or
                np.linalg.norm(q) < 1e-8 or not 0 <= opacity <= 1):
            raise ValueError("invalid Gaussian attributes")
        if "sh" in item:
            coefficients = np.asarray(item["sh"])
            if coefficients.shape not in ((1,3), (4,3)):
                raise ValueError("invalid spherical harmonics")
            numbers(coefficients, coefficients.shape)
    return model


def scene_at(model, t):
    """A static scene at t seconds; no camera is involved in this operation."""
    validate_model(model)
    if not 0 <= t <= model["duration"]:
        raise ValueError("time outside the clip")
    scene = deepcopy(model["canonical"])
    weight = pulse(t/model["duration"])
    for spot, amplitude in zip(scene["splats"], model["amplitudes"]):
        spot["center"] = (np.asarray(spot["center"]) + weight*np.asarray(amplitude)).tolist()
    return scene


def reference_scene(t):
    """Synthetic film. A small asymmetric sway is outside the fitted pulse family."""
    if not 0 <= t <= 1:
        raise ValueError("time outside the reference clip")
    scene = ribbon()
    for index, spot in enumerate(scene["splats"]):
        s = index/4
        travel = np.array([.12*s, -.8*s*s, .45*s])*pulse(t)
        sway = np.array([0., .12*s, .08*s])*np.sin(2*np.pi*t)
        spot["center"] = (np.asarray(spot["center"]) + travel + sway).tolist()
    return scene


def stereo_tracks(noise=0.):
    """Labeled center pixels, not RGB targets. Only the two training times enter."""
    observations = []
    for time_index, t in enumerate(TRAIN_TIMES):
        left, right = [], []
        for i, spot in enumerate(reference_scene(t)["splats"]):
            x, y, z = spot["center"]
            u = PRINCIPAL[0] + FOCAL*x/z
            v = PRINCIPAL[1] + FOCAL*y/z
            # A repeatable horizontal annotation error in the right image, in pixels.
            error = noise*(-1)**(time_index+i)
            left.append([u, v])
            right.append([u - FOCAL*BASELINE/z + error, v])
        observations.append(dict(t=t, left=left, right=right))
    return observations


def fit_motion(canonical, observations):
    """Least squares in world space after triangulating labeled stereo centers."""
    base = np.array([s["center"] for s in canonical["splats"]])
    numerator = np.zeros_like(base, dtype=float)
    denominator = 0.
    for row in observations:
        left, right = np.asarray(row["left"]), np.asarray(row["right"])
        if left.shape != (len(base), 2) or right.shape != left.shape or not 0 <= row["t"] <= 1:
            raise ValueError("invalid stereo observation")
        disparity = left[:, 0]-right[:, 0]
        if not np.isfinite(disparity).all() or np.any(disparity <= 0):
            raise ValueError("positive disparity required")
        z = FOCAL*BASELINE/disparity
        x = (left[:, 0]-PRINCIPAL[0])*z/FOCAL
        y = (left[:, 1]-PRINCIPAL[1])*z/FOCAL
        h = pulse(row["t"])
        numerator += h*(np.column_stack((x, y, z))-base)
        denominator += h*h
    if denominator <= 0:
        raise ValueError("at least one observation must be inside the time interval")
    return validate_model(dict(format=FORMAT, duration=1., canonical=deepcopy(canonical),
                               amplitudes=(numerator/denominator).tolist()))


def example_model(noise=0.):
    return fit_motion(ribbon(), stereo_tracks(noise))
