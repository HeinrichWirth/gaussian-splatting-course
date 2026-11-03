"""A small, exact (no screen-space truncation) front-to-back splat compositor."""
from copy import deepcopy
import math

import numpy as np

from geometry3d import covariance_3d, project_point, projected_covariance, world_to_camera


FORMAT = "gs-course-3d-v1"
C0 = 1 / math.sqrt(4 * math.pi)
C1 = math.sqrt(3 / (4 * math.pi))


def splat(center, scales, color, opacity, rotation=(1, 0, 0, 0)):
    return dict(center=list(center), scales=list(scales), rotation=list(rotation),
                color=list(color), opacity=float(opacity))


def make_scene(*splats):
    return {"format": FORMAT, "splats": list(splats)}


def orbit_camera(angle=0, focal=48, target=(0, 0, 4), radius=4):
    """Orbit about world Y. At zero degrees, right/down/forward match world axes."""
    a = math.radians(angle)
    co, si = math.cos(a), math.sin(a)
    rotation = np.array([[co, 0, si], [0, 1, 0], [-si, 0, co]])
    return dict(center=(np.asarray(target) - radius * rotation[:, 2]).tolist(),
                rotation=rotation.tolist(), focal=focal)


def sh_basis(direction):
    d = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if d.shape != (3,) or not np.isfinite(norm) or norm == 0:
        raise ValueError("a nonzero finite direction is required")
    x, y, z = d / norm
    return np.array([C0, -C1*y, C1*z, -C1*x])


def sh_color(direction, coefficients):
    coeff = np.asarray(coefficients, dtype=float)
    if coeff.shape not in ((1, 3), (4, 3)) or not np.all(np.isfinite(coeff)):
        raise ValueError("expected one or four RGB coefficient triplets")
    return np.clip(.5 + sh_basis(direction)[:len(coeff)] @ coeff, 0, 1)


def project_splats(scene, camera, width, height):
    if scene.get("format") != FORMAT:
        raise ValueError("unsupported 3D scene format")
    camera = camera or {}
    R = np.asarray(camera.get("rotation", np.eye(3)), dtype=float)
    origin = np.asarray(camera.get("center", [0, 0, 0]), dtype=float)
    focal = camera.get("focal", min(width, height))
    principal = camera.get("principal", ((width-1)/2, (height-1)/2))
    near = camera.get("near", .05)
    projected = []
    for index, item in enumerate(scene["splats"]):
        mean = world_to_camera(item["center"], origin, R)
        if mean[2] <= near:
            continue
        cov = covariance_3d(item["scales"], item["rotation"])
        screen_cov = projected_covariance(mean, R.T @ cov @ R, focal)
        color = (sh_color(np.asarray(item["center"]) - origin, item["sh"])
                 if "sh" in item else np.asarray(item["color"], dtype=float))
        if color.shape != (3,) or not np.all((color >= 0) & (color <= 1)):
            raise ValueError("RGB must lie in [0,1]")
        if not 0 <= item["opacity"] <= 1:
            raise ValueError("opacity must lie in [0,1]")
        projected.append(dict(index=index, depth=mean[2], color=color,
                              mean=project_point(mean, focal, principal, near),
                              opacity=item["opacity"], covariance=screen_cov,
                              precision=np.linalg.inv(screen_cov)))
    return sorted(projected, key=lambda p: p["depth"])


def footprint_alpha(pixel_xy, item):
    """One pixel or a grid of XY pairs, integer positions are pixel centers."""
    delta = np.asarray(pixel_xy) - item["mean"]
    q = np.einsum("...i,ij,...j->...", delta, item["precision"], delta)
    return item["opacity"] * np.exp(-.5 * q)


def render_scene(scene, camera=None, width=65, height=65, background=(1, 1, 1),
                 reverse=False):
    spots = project_splats(scene, camera, width, height)
    yy, xx = np.indices((height, width))
    pixels = np.stack((xx, yy), axis=-1)
    rgb = np.zeros((height, width, 3))
    remaining = np.ones((height, width))
    for item in reversed(spots) if reverse else spots:
        alpha = footprint_alpha(pixels, item)
        rgb += (remaining * alpha)[..., None] * item["color"]
        remaining *= 1 - alpha
    return rgb + remaining[..., None] * np.asarray(background)


def overlap_scene(red_depth=3.5):
    return make_scene(splat((0, 0, red_depth), (.55, .35, .25), (1, 0, 0), .6),
                      splat((0, 0, 4.5), (.38, .65, .22), (0, 0, 1), .5))


def ray_scene(depth):
    """All lengths scale together about the first camera, fixing its image."""
    return make_scene(splat((.10*depth, -.05*depth, depth),
                            (.12*depth, .08*depth, .06*depth), (.85, .2, .08), .9))


def translated_camera(x, focal=28):
    return {"center": [x, 0, 0], "rotation": np.eye(3).tolist(), "focal": focal}


def ray_loss(depth, baselines=(0., .8), size=33):
    losses = []
    for x in baselines:
        camera = translated_camera(x)
        image = render_scene(ray_scene(depth), camera, size, size)
        target = render_scene(ray_scene(4), camera, size, size)
        losses.append(np.mean((image - target)**2))
    return float(np.mean(losses))


def depth_step(depth, baselines=(0., .8), rate=80., h=.002):
    gradient = (ray_loss(depth+h, baselines) - ray_loss(depth-h, baselines))/(2*h)
    return float(np.clip(depth - rate*gradient, 1.25, 7)), gradient


def density_scene(target=False):
    color = (.12, .37, .83)
    if target:
        return make_scene(*(splat((x, 0, 4), (.23, .34, .16), color, .9)
                            for x in (-.54, .42)))
    return make_scene(splat((-.16, 0, 4), (.63, .38, .18), color, .77),
                      splat((.78, -.55, 4.2), (.20, .18, .13), color, .015))


def density_loss(scene, size=33):
    camera = translated_camera(0, focal=42)
    image = render_scene(scene, camera, size, size)
    target = render_scene(density_scene(True), camera, size, size)
    return float(np.mean((image-target)**2))


def density_gradients(scene, h=.0001):
    """Centered differences for center X and opacity, only these two are trained."""
    result = []
    for i, item in enumerate(scene["splats"]):
        derivatives = []
        for field in ("x", "opacity"):
            samples = []
            value = item["center"][0] if field == "x" else item["opacity"]
            bounds = (value-h, value+h) if field == "x" else (max(0, value-h), min(1, value+h))
            for sample in bounds:
                candidate = deepcopy(scene)
                spot = candidate["splats"][i]
                if field == "x":
                    spot["center"][0] = sample
                else:
                    spot["opacity"] = sample
                samples.append(density_loss(candidate))
            derivatives.append((samples[1]-samples[0])/(bounds[1]-bounds[0]))
        result.append(derivatives)
    return result


def density_step(scene, rate=3):
    updated = deepcopy(scene)
    for item, (dx, do) in zip(updated["splats"], density_gradients(scene)):
        item["center"][0] = float(np.clip(item["center"][0] - rate*dx, -2, 2))
        item["opacity"] = float(np.clip(item["opacity"] - rate*do, 0, 1))
    return updated


def density_edit(scene, operation, selected=0):
    """A deterministic horizontal split for the axis-aligned lesson-17 scene."""
    updated = deepcopy(scene)
    items = updated["splats"]
    if operation == "prune":
        updated["splats"] = [item for item in items if item["opacity"] >= .03]
    elif operation == "reset":
        for item in items:
            item["opacity"] = min(.01, item["opacity"])
    elif operation in ("clone", "split"):
        if not 0 <= selected < len(items):
            raise ValueError("a spot must be selected")
        if len(items) >= 16:
            raise ValueError("the lesson scene supports at most 16 splats")
        item = items[selected]
        if operation == "clone":
            items.append(deepcopy(item))
        else:
            children = []
            for sign in (-1, 1):
                child = deepcopy(item)
                child["center"][0] += sign*.65*item["scales"][0]
                child["scales"] = [s/1.6 for s in item["scales"]]
                children.append(child)
            items[selected:selected+1] = children
    else:
        raise ValueError("unknown density operation")
    return updated


def density_pass(scene, threshold=.001):
    """One teaching policy: inspect the original scene once, then edit indices backwards."""
    gradients = density_gradients(scene)
    updated = deepcopy(scene)
    for i in reversed(range(len(gradients))):
        spot = scene["splats"][i]
        if spot["opacity"] < .03:
            del updated["splats"][i]
        elif abs(gradients[i][0]) > threshold and len(updated["splats"]) < 16:
            op = "split" if spot["scales"][0] > .32 else "clone"
            updated = density_edit(updated, op, i)
    return updated
