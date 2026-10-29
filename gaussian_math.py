"""Small numerical building blocks for the plane experiments in lessons 3–5.

Coordinates use a right-handed drawing plane: x right, y up. Angles are degrees,
positive counterclockwise. Scales belong to the Gaussian's own axes. A Gaussian
weight has peak 1 and is not a normalized probability density. Layer lists are
ordered front to back; RGB values are the lesson's numeric channels in [0, 1].
"""

from __future__ import annotations

import json
import math

import numpy as np


def _scalar(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _pair(value, name, positive=False):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two finite numbers") from exc
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain two finite numbers")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return result.copy()


def _color(value):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("RGB must contain three finite numbers between 0 and 1") from exc
    if (result.shape != (3,) or not np.all(np.isfinite(result))
            or np.any(result < 0) or np.any(result > 1)):
        raise ValueError("RGB must contain three finite numbers between 0 and 1")
    return result.copy()


def rotation_matrix(angle_deg):
    """Rotate a column vector counterclockwise in the x/y drawing plane."""
    angle = math.radians(_scalar(angle_deg, "angle_deg"))
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)


def covariance_matrix(sx=2.0, sy=1.0, angle_deg=0.0):
    """Shape matrix R diag(sx², sy²) Rᵀ; sx and sy are local-axis scales."""
    scales = _pair((sx, sy), "scales", positive=True)
    rotation = rotation_matrix(angle_deg)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        variances = scales * scales
        matrix = (rotation * variances) @ rotation.T
    if not np.all(np.isfinite(matrix)) or np.any(variances == 0):
        raise ValueError("scales are outside the supported floating-point range")
    return matrix


def inverse_covariance(matrix):
    """Invert a symmetric positive-definite 2×2 matrix explicitly.

    For [[a, b], [b, c]], the determinant is a*c-b*b, and the inverse is
    [[c, -b], [-b, a]] / determinant. Singular and indefinite shapes are rejected.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be a finite 2 by 2 matrix")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    a, c = float(matrix[0, 0]), float(matrix[1, 1])
    b = float((matrix[0, 1] + matrix[1, 0]) / 2)
    determinant = a * c - b * b
    if a <= 0 or determinant <= 0 or not math.isfinite(determinant):
        raise ValueError("covariance must be positive definite and invertible")
    inverse = np.array([[c, -b], [-b, a]]) / determinant
    if not np.all(np.isfinite(inverse)):
        raise ValueError("inverse is outside the supported floating-point range")
    return inverse


def gaussian2d(x, y, mux=0.0, muy=0.0, sx=2.0, sy=1.0, angle_deg=0.0):
    """Evaluate a peak-one Gaussian; x and y can be broadcastable arrays."""
    center = _pair((mux, muy), "center")
    scales = _pair((sx, sy), "scales", positive=True)
    rotation = rotation_matrix(angle_deg)
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("coordinates must be finite")
    dx, dy = x - center[0], y - center[1]
    local_x = rotation[0, 0] * dx + rotation[1, 0] * dy
    local_y = rotation[0, 1] * dx + rotation[1, 1] * dy
    with np.errstate(over="ignore", under="ignore"):
        q = (local_x / scales[0]) ** 2 + (local_y / scales[1]) ** 2
        return np.exp(-0.5 * q)


def gaussian2d_steps(x=2.0, y=1.0, mux=0.0, muy=0.0,
                     sx=2.0, sy=1.0, angle_deg=0.0):
    """Expose every operation for a single point, plus equivalent matrices."""
    x, y = _scalar(x, "x"), _scalar(y, "y")
    center = _pair((mux, muy), "center")
    scales = _pair((sx, sy), "scales", positive=True)
    rotation = rotation_matrix(angle_deg)
    c, s = float(rotation[0, 0]), float(rotation[1, 0])
    dx, dy = x - center[0], y - center[1]
    local_x, local_y = c * dx + s * dy, -s * dx + c * dy
    ux, uy = local_x / scales[0], local_y / scales[1]
    q = ux * ux + uy * uy
    exponent = -q / 2
    covariance = covariance_matrix(*scales, angle_deg)
    inverse = inverse_covariance(covariance)
    return dict(dx=float(dx), dy=float(dy), cos=c, sin=s,
                local_x=float(local_x), local_y=float(local_y),
                ux=float(ux), uy=float(uy), q=float(q), exponent=float(exponent),
                weight=math.exp(exponent), covariance=covariance, inverse=inverse)


def make_layer(name="Spot", center=(0, 0), scales=(1, 1), angle_deg=0,
               color=(1, 0, 0), opacity=0.5):
    """Create an independent, JSON-compatible Gaussian layer."""
    if not isinstance(name, str) or not name.strip() or len(name) > 80:
        raise ValueError("layer name must contain 1 to 80 characters")
    opacity = _scalar(opacity, "opacity")
    if not 0 <= opacity <= 1:
        raise ValueError("opacity must be between 0 and 1")
    scales = _pair(scales, "scales", positive=True)
    angle_deg = _scalar(angle_deg, "angle_deg")
    # The editor and its matrix display must agree on valid shapes.
    inverse_covariance(covariance_matrix(*scales, angle_deg))
    return dict(name=name, center=_pair(center, "center").tolist(),
                scales=scales.tolist(), angle_deg=angle_deg,
                color=_color(color).tolist(), opacity=opacity)


def _layers(layers):
    if not isinstance(layers, (list, tuple)) or len(layers) > 256:
        raise ValueError("scene must be a list with at most 256 layers")
    result = []
    fields = {"name", "center", "scales", "angle_deg", "color", "opacity"}
    for layer in layers:
        if not isinstance(layer, dict) or set(layer) != fields:
            raise ValueError("each layer must contain name, center, scales, angle_deg, color and opacity")
        result.append(make_layer(**layer))
    return result


def composite_at(x, y, layers, background=(1, 1, 1)):
    """Trace a single pixel, visiting layers from front to back.

    Each contribution is T_before * alpha * color. The background receives
    the final T, so the layer weights and background weight sum to one.
    """
    x, y = _scalar(x, "x"), _scalar(y, "y")
    layers, background = _layers(layers), _color(background)
    color = np.zeros(3)
    transmittance = 1.0
    trace = []
    for layer in layers:
        weight = float(gaussian2d(x, y, *layer["center"], *layer["scales"], layer["angle_deg"]))
        alpha = layer["opacity"] * weight
        before = transmittance
        contribution = before * alpha * np.asarray(layer["color"])
        color += contribution
        transmittance *= 1 - alpha
        trace.append(dict(name=layer["name"], weight=weight, alpha=alpha,
                          transmittance_before=before, contribution=contribution,
                          transmittance_after=transmittance))
    background_contribution = transmittance * background
    return dict(color=color + background_contribution, transmittance=transmittance,
                layers=trace, background_contribution=background_contribution)


def render_layers(layers, background=(1, 1, 1), size=161, bounds=(-4, 4, -4, 4)):
    """Sample the same composition on an inclusive grid, y increasing upward.

    Use origin='lower' when displaying. No gamma conversion or opacity cutoff
    is applied: the displayed RGB channels match the lesson's arithmetic.
    """
    layers, background = _layers(layers), _color(background)
    if isinstance(size, bool) or int(size) != size or not 2 <= size <= 1025:
        raise ValueError("size must be an integer from 2 to 1025")
    size = int(size)
    bounds = np.asarray(bounds, dtype=float)
    if (bounds.shape != (4,) or not np.all(np.isfinite(bounds))
            or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]):
        raise ValueError("bounds must be xmin, xmax, ymin, ymax with increasing limits")
    x, y = np.meshgrid(np.linspace(bounds[0], bounds[1], size),
                       np.linspace(bounds[2], bounds[3], size))
    image = np.zeros((size, size, 3))
    transmittance = np.ones((size, size))
    for layer in layers:
        alpha = layer["opacity"] * gaussian2d(x, y, *layer["center"],
                                               *layer["scales"], layer["angle_deg"])
        image += (transmittance * alpha)[..., None] * np.asarray(layer["color"])
        transmittance *= 1 - alpha
    image += transmittance[..., None] * background
    return np.clip(image, 0, 1)


def scene_json(layers, background=(1, 1, 1)):
    """Serialize the editable parameters, not an image or widget session."""
    scene = dict(format="gaussian-splatting-course-2d", version=1,
                 background=_color(background).tolist(), layers=_layers(layers))
    return json.dumps(scene, ensure_ascii=False, indent=2, allow_nan=False)


def load_scene_json(text):
    """Validate a saved scene and return independent (layers, background)."""
    if not isinstance(text, str) or len(text) > 1_000_000:
        raise ValueError("scene JSON must be text shorter than one million characters")
    scene = json.loads(text)
    if (not isinstance(scene, dict) or scene.get("format") != "gaussian-splatting-course-2d"
            or type(scene.get("version")) is not int or scene["version"] != 1):
        raise ValueError("unsupported scene format or version")
    if "layers" not in scene or "background" not in scene:
        raise ValueError("scene needs layers and background")
    return _layers(scene["layers"]), _color(scene["background"]).tolist()
