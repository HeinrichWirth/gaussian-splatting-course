"""Small, inspectable PyTorch experiments for lessons 8 and 9.

Coordinates and compositing match gaussian_math.py: x right, y up, layers
front to back, white background, numeric RGB in [0, 1]. The image-fitting
model keeps angles at zero and trains centers, scales, RGB, and opacity.
Torch is imported only when a training function is used.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np

from gaussian_math import make_layer, scene_json

BOUNDS = (-4.0, 4.0, -4.0, 4.0)
RATES = {"centers": 4.0, "scales": 4.0, "colors": 4.0, "opacity": 4.0}


def _integer(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not np.isfinite(value) or int(value) != value
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _torch():
    import torch
    return torch


@contextmanager
def small_cpu_work():
    """Avoid thread scheduling overhead for tiny tensors; restore global state."""
    torch = _torch()
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def chain_example(mu=0.0, x=1.0, sigma=1.0, target=.8):
    if not np.isfinite([mu, x, sigma, target]).all() or sigma <= 0:
        raise ValueError("Finite inputs and a positive scale are required")
    d = x - mu
    u = d / sigma
    q = u * u
    t = -q / 2
    weight = float(np.exp(t))
    residual = weight - target
    loss = residual * residual
    local = [2 * residual, weight, -.5, 2 * u, 1 / sigma, -1.0]
    return dict(d=d, u=u, q=q, t=t, weight=weight, residual=residual,
                loss=loss, local=local, gradient=float(np.prod(local)))


def autograd_check(mu=0.0, epsilon=1e-5):
    """Compare the full backward chain, PyTorch, and a central difference."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    torch = _torch()
    parameter = torch.tensor(float(mu), dtype=torch.float64, requires_grad=True)
    weight = torch.exp(-.5 * (1 - parameter).square())
    loss = (weight - .8).square()
    loss.backward()
    finite = (chain_example(mu + epsilon)["loss"] -
              chain_example(mu - epsilon)["loss"]) / (2 * epsilon)
    return dict(chain=chain_example(mu)["gradient"],
                autograd=parameter.grad.item(), finite_difference=finite)


def fit_one_gaussian(initial=-1.0, target_center=1.5, rate=2.0, steps=80):
    """Full-batch SGD for one center, with fixed unit scale and 41 samples."""
    torch = _torch()
    if not np.isfinite([initial, target_center, rate]).all() or rate < 0:
        raise ValueError("Finite values and a nonnegative rate are required")
    steps = _integer(steps, "steps", 0, 2000)
    x = torch.linspace(-4, 4, 41, dtype=torch.float64)
    target = torch.exp(-.5 * (x - target_center).square())
    center = torch.tensor(float(initial), dtype=torch.float64, requires_grad=True)
    history = []
    with small_cpu_work():
        for step in range(int(steps) + 1):
            prediction = torch.exp(-.5 * (x - center).square())
            loss = (prediction - target).square().mean()
            history.append(dict(step=step, center=center.item(), loss=loss.item()))
            if step == steps:
                break
            loss.backward()
            with torch.no_grad():
                center -= rate * center.grad
            center.grad = None
    return dict(x=x.numpy(), target=target.numpy(),
                prediction=prediction.detach().numpy(), history=history)


def make_target(name="shapes", size=32):
    """Deterministic RGB targets, bottom row first, without external downloads."""
    size = _integer(size, "size", 8, 256)
    if name == "flower":
        from mixing_demos import preset_scene
        from gaussian_math import render_layers
        layers, background = preset_scene("flower", "en")
        return render_layers(layers, background, size=size, bounds=BOUNDS)
    axis = np.linspace(-4, 4, size)
    x, y = np.meshgrid(axis, axis)
    image = np.ones((size, size, 3))
    if name == "shapes":
        image[((x + 1.15) ** 2 + (y - 1.0) ** 2) < 1.45 ** 2] = [.95, .32, .12]
        image[(x > .1) & (x < 2.9) & (y > -2.7) & (y < -.3)] = [.18, .36, .85]
        image[(np.abs(y - .65 * x - 1.1) < .27) & (x > -.6) & (x < 3)] = [.12, .58, .27]
    elif name == "checker":
        check = (np.floor((x + 4) * 1.5) + np.floor((y + 4) * 1.5)) % 2 == 0
        inside = (np.abs(x) < 3) & (np.abs(y) < 3)
        image[inside & check] = [.2, .27, .65]
        image[inside & ~check] = [.95, .62, .15]
    else:
        raise ValueError("target must be shapes, flower, or checker")
    return image


def load_image(path, size=32):
    """Fit a local image inside a white square, preserving its aspect ratio.

    File rows go downward; renderer rows go upward, hence the final flip.
    Transparency is composed onto white before RGB conversion.
    """
    from PIL import Image, ImageOps
    size = _integer(size, "size", 8, 256)
    with Image.open(Path(path)) as source:
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
        rgba.thumbnail((int(size), int(size)), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (int(size), int(size)), "white")
        canvas.alpha_composite(rgba, ((size - rgba.width) // 2, (size - rgba.height) // 2))
        return np.flipud(np.asarray(canvas.convert("RGB"), dtype=float) / 255).copy()


def _target(array):
    image = np.asarray(array, dtype=float)
    if (image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 2
            or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1):
        raise ValueError("target must be a finite H × W × 3 RGB array in [0, 1]")
    return image


def initialize(target, count=64, seed=7, placement="grid", scale=.55):
    """Initialize independent, trainable tensors; sample colors from the target."""
    torch = _torch()
    target = _target(target)
    count = _integer(count, "count", 1, 128)
    if not .2 <= scale <= 3:
        raise ValueError("scale must be in [.2, 3]")
    rng = np.random.default_rng(seed)
    if placement == "grid":
        side = int(np.ceil(np.sqrt(count)))
        axis = -4 + (np.arange(side) + .5) * 8 / side
        xx, yy = np.meshgrid(axis, axis[::-1])
        centers = np.column_stack((xx.ravel(), yy.ravel()))[:count]
    elif placement == "random":
        centers = rng.uniform(-3.7, 3.7, (count, 2))
    elif placement == "center":
        centers = rng.normal(0, .35, (count, 2))
    else:
        raise ValueError("placement must be grid, random, or center")
    rows = np.rint((centers[:, 1] + 4) / 8 * (target.shape[0] - 1)).astype(int)
    cols = np.rint((centers[:, 0] + 4) / 8 * (target.shape[1] - 1)).astype(int)
    rows = np.clip(rows, 0, target.shape[0] - 1)
    cols = np.clip(cols, 0, target.shape[1] - 1)
    # A little white in each sampled color avoids exact-boundary starts.
    colors = .9 * target[rows, cols] + .1
    values = dict(centers=centers, scales=np.full((count, 2), float(scale)),
                  colors=colors, opacity=np.full(count, .65))
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True)
            for key, value in values.items()}


def render_torch(parameters, height=32, width=None):
    """Differentiable front-to-back alpha compositing with a fixed layer order."""
    torch = _torch()
    width = height if width is None else width
    ref = parameters["centers"]
    x = torch.linspace(-4, 4, width, dtype=ref.dtype, device=ref.device)
    y = torch.linspace(-4, 4, height, dtype=ref.dtype, device=ref.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    dx = (xx[None] - parameters["centers"][:, 0, None, None])
    dy = (yy[None] - parameters["centers"][:, 1, None, None])
    q = (dx / parameters["scales"][:, 0, None, None]).square()
    q = q + (dy / parameters["scales"][:, 1, None, None]).square()
    alpha = parameters["opacity"][:, None, None] * torch.exp(-.5 * q)
    remaining = torch.cumprod(1 - alpha, dim=0)
    before = torch.cat((torch.ones_like(remaining[:1]), remaining[:-1]), dim=0)
    foreground = (before * alpha)[..., None] * parameters["colors"][:, None, None, :]
    return foreground.sum(dim=0) + remaining[-1, ..., None]


def projected_step(parameters, rates=None):
    """Ordinary SGD, followed by explicit parameter bounds; no hidden optimizer."""
    torch = _torch()
    rates = RATES if rates is None else rates
    if set(parameters) != set(RATES) or set(rates) != set(parameters):
        raise ValueError("all four parameter groups and their rates are required")
    for name, value in parameters.items():
        if not np.isscalar(rates[name]) or not np.isfinite(rates[name]) or rates[name] < 0:
            raise ValueError("learning rates must be finite and nonnegative")
        if value.grad is None or not torch.isfinite(value.grad).all():
            raise ValueError("finite gradients from backward are required before an update")
    with torch.no_grad():
        for name, value in parameters.items():
            value -= rates[name] * value.grad
        parameters["centers"].clamp_(-4, 4)
        parameters["scales"].clamp_(.2, 3)
        parameters["colors"].clamp_(0, 1)
        parameters["opacity"].clamp_(0, 1)
    for value in parameters.values():
        value.grad = None


def parameters_scene(parameters):
    arrays = {name: value.detach().cpu().numpy() for name, value in parameters.items()}
    layers = [make_layer(f"Gaussian {i + 1}", center=arrays["centers"][i],
                         scales=arrays["scales"][i], angle_deg=0,
                         color=arrays["colors"][i], opacity=arrays["opacity"][i])
              for i in range(len(arrays["centers"]))]
    return scene_json(layers)


def fit_image(target=None, count=64, steps=120, seed=7, placement="grid",
              scale=.55, rate_multiplier=1.0, snapshot_steps=None):
    """Train a small image model and retain only useful progress snapshots."""
    torch = _torch()
    target = _target(make_target() if target is None else target)
    steps = _integer(steps, "steps", 0, 2000)
    if not np.isfinite(rate_multiplier) or not 0 <= rate_multiplier <= 20:
        raise ValueError("rate_multiplier must be in [0, 20]")
    parameters = initialize(target, count, seed, placement, scale)
    target_tensor = torch.tensor(target, dtype=torch.float64)
    rates = {k: v * rate_multiplier for k, v in RATES.items()}
    keep = set(snapshot_steps or [0, int(steps) // 4, int(steps) // 2, int(steps)])
    history, snapshots = [], {}
    with small_cpu_work():
        for step in range(int(steps) + 1):
            prediction = render_torch(parameters, *target.shape[:2])
            loss = (prediction - target_tensor).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; reduce the update size")
            history.append(loss.item())
            if step in keep:
                snapshots[step] = prediction.detach().numpy().copy()
            if step == steps:
                break
            loss.backward()
            projected_step(parameters, rates)
    return dict(target=target, image=prediction.detach().numpy(), history=history,
                snapshots=snapshots, parameters=parameters, scene=parameters_scene(parameters),
                settings=dict(count=count, steps=int(steps), seed=seed,
                              placement=placement, scale=scale, rates=rates))
