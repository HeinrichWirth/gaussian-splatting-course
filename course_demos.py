"""Offline, bilingual Jupyter experiments for the first two course lessons.

All figure functions return closed Matplotlib figures: use ``display(figure)``.
The pure numerical functions are independent of Jupyter and can be tested directly.
"""

from __future__ import annotations

import math
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Wedge


BLUE = "#2463a9"
ORANGE = "#d85b27"
GREEN = "#23866c"
PURPLE = "#8256a9"
INK = "#26364a"
MUTED = "#52647a"
PAPER = "#f4f7fb"


def _text(language, ru, en):
    if language not in ("ru", "en"):
        raise ValueError("language must be 'ru' or 'en'")
    return ru if language == "ru" else en


def _finish(fig):
    plt.close(fig)
    return fig


def _axes(ax):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#b6c3d1")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(alpha=0.16, color=MUTED)
    ax.set_axisbelow(True)


def _number(value, digits=3):
    if abs(value) < 5 * 10 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") or "0"


def _weight(value):
    """Keep tiny positive weights visibly distinct from floating-point zero."""
    return f"{value:.3e}" if 0 < value < 1e-6 else f"{value:.6f}"


def gaussian_steps(x=4.0, mu=2.0, sigma=2.0):
    """Return the five scalar steps of an unnormalized 1D Gaussian.

    Inputs are finite real coordinates and a strictly positive scale. The output
    weight is dimensionless. Very small floating-point weights may underflow to 0.
    """
    x, mu, sigma = float(x), float(mu), float(sigma)
    if not all(math.isfinite(v) for v in (x, mu, sigma)):
        raise ValueError("x, mu and sigma must be finite")
    if sigma <= 0:
        raise ValueError("sigma must be strictly positive")
    offset = x - mu
    standardized = offset / sigma
    squared = standardized * standardized
    exponent = -squared / 2
    weight = math.exp(exponent)
    return dict(offset=offset, standardized=standardized, squared=squared,
                exponent=exponent, weight=weight)


def gaussian_values(x, mu=2.0, sigma=2.0):
    """Evaluate the same unnormalized weight at an array of coordinates."""
    gaussian_steps(mu, mu, sigma)
    coordinates = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must be finite")
    return np.exp(-0.5 * ((coordinates - mu) / sigma) ** 2)


def gaussian_figure(mu=2.0, sigma=2.0, x=4.0, language="ru"):
    """Fixed-coordinate curve, brightness strip and complete scalar calculation."""
    steps = gaussian_steps(x, mu, sigma)
    coordinates = np.linspace(-12, 12, 1201)
    weights = gaussian_values(coordinates, mu, sigma)
    fig = plt.figure(figsize=(10.8, 8.0), layout="constrained", facecolor=PAPER)
    grid = fig.add_gridspec(3, 1, height_ratios=[3.6, 0.7, 2.6])
    ax = fig.add_subplot(grid[0])
    _axes(ax)
    ax.fill_between(coordinates, weights, color=BLUE, alpha=0.09)
    ax.plot(coordinates, weights, color=BLUE, lw=2.6)
    for k in range(-3, 4):
        location = mu + k * sigma
        if -12 <= location <= 12:
            height = math.exp(-0.5 * k * k)
            ax.vlines(location, 0, height, color=MUTED, alpha=0.35, ls=":")
            ax.scatter([location], [height], s=19, color=BLUE, zorder=4)
            label = "μ" if k == 0 else f"μ{'+' if k > 0 else '−'}{abs(k) if abs(k) != 1 else ''}σ"
            if sigma >= 0.55 or k == 0:
                ax.text(location, 0.98, label, color=MUTED, ha="center", va="top", fontsize=8)
    ax.axvline(x, color=ORANGE, alpha=0.75, ls="--", lw=1.2)
    ax.scatter([x], [steps["weight"]], color=ORANGE, edgecolor="white", s=85, zorder=6)
    ax.set(xlim=(-12, 12), ylim=(0, 1.05), xlabel=_text(language, "Координата x", "Coordinate x"),
           ylabel=_text(language, "Вес f(x)", "Weight f(x)"))
    ax.set_title(_text(language, "Один колокол: центр, масштаб и выбранная точка",
                       "One bell: center, scale and a selected point"), loc="left", color=INK, fontsize=14, pad=14)
    ax.text(0.99, 0.82, f"μ = {_number(mu)}    σ = {_number(sigma)}\nx = {_number(x)}    f(x) = {_weight(steps['weight'])}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color=INK,
            bbox=dict(facecolor="white", edgecolor="#dce5ee", boxstyle="round,pad=0.6"))
    strip = fig.add_subplot(grid[1])
    strip.imshow(weights[None, :], extent=[-12, 12, 0, 1], aspect="auto", origin="lower",
                 cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    strip.axvline(x, color="#5eead4", lw=1.5)
    strip.set_yticks([])
    strip.set_xticks([-12, -6, 0, 6, 12])
    strip.tick_params(labelsize=9)
    strip.set_title(_text(language, "Те же значения как яркость: тёмное = 0, светлое = 1",
                          "The same values as brightness: dark = 0, light = 1"), fontsize=10, loc="left", color=INK)
    calc = fig.add_subplot(grid[2])
    calc.axis("off")
    names = _text(language,
                  ["1. Смещение", "2. В единицах масштаба", "3. Квадрат", "4. Показатель степени", "5. Вес"],
                  ["1. Offset", "2. In units of scale", "3. Square", "4. Exponent", "5. Weight"])
    d, z, q, a, w = steps.values()
    calculations = [f"{_number(x)} − ({_number(mu)}) = {_number(d)}",
                    f"{_number(d)} / {_number(sigma)} = {_number(z)}",
                    f"({_number(z)}) × ({_number(z)}) = {_number(q)}",
                    f"−({_number(q)}) / 2 = {_number(a)}",
                    f"exp({_number(a)}) = {_weight(w)}"]
    table = calc.table(cellText=list(zip(names, calculations)), colWidths=[0.37, 0.63],
                       cellLoc="left", bbox=[0, 0.16, 1, 0.84])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dce5ee")
        cell.set_facecolor("white" if row % 2 == 0 else "#edf3f8")
        cell.set_text_props(color=INK)
    calc.text(0, 0.03, _text(language, "На расстоянии σ: 0.606531  ·  2σ: 0.135335  ·  3σ: 0.011109 (не ноль)",
                            "At distance σ: 0.606531  ·  2σ: 0.135335  ·  3σ: 0.011109 (not zero)"),
              fontsize=10, color=MUTED, transform=calc.transAxes)
    return _finish(fig)


def construction_figure(language="ru"):
    """The five operations, each drawn against the original x coordinate."""
    coordinates = np.linspace(-4, 8, 601)
    mu, sigma, x = 2, 2, 4
    offset = coordinates - mu
    normalized = offset / sigma
    stages = [offset, normalized, normalized ** 2, -normalized ** 2 / 2,
              np.exp(-normalized ** 2 / 2)]
    selected = [2, 1, 1, -0.5, math.exp(-0.5)]
    titles = _text(language,
                   ["1. Вычитаем центр", "2. Делим на масштаб", "3. Возводим в квадрат", "4. Умножаем на −1/2", "5. Применяем exp"],
                   ["1. Subtract the center", "2. Divide by the scale", "3. Square", "4. Multiply by −1/2", "5. Apply exp"])
    formulas = ["x − μ", "(x − μ) / σ", "((x − μ) / σ)²", "−((x − μ) / σ)² / 2", "exp(−((x − μ) / σ)² / 2)"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.3), layout="constrained", facecolor=PAPER)
    for ax, values, value, title, formula in zip(axes.flat, stages, selected, titles, formulas):
        _axes(ax)
        ax.plot(coordinates, values, color=BLUE, lw=2)
        ax.axhline(0, color=MUTED, lw=0.7, alpha=0.5)
        ax.axvline(mu, color=MUTED, ls=":", alpha=0.5)
        ax.axvline(x, color=ORANGE, ls="--", alpha=0.6)
        ax.scatter([x], [value], color=ORANGE, zorder=4, s=40)
        ax.set_title(title + "\n" + formula, fontsize=10, color=INK, pad=9)
        ax.set_xlabel("x", color=MUTED, fontsize=9)
        ax.text(0.03, 0.96, f"x = 4 → {_number(value, 6)}", transform=ax.transAxes,
                fontsize=9, color=INK, va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
        ax.set_xlim(-4, 8)
    # Keeping these axes identical makes division by sigma visibly change slope.
    axes.flat[0].set_ylim(-6.6, 6.6)
    axes.flat[1].set_ylim(-6.6, 6.6)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.04, 0.85,
                       _text(language, "Одно x проходит\nвсю цепочку.\n\nμ = 2, σ = 2, x = 4\n\n2 → 1 → 1 → −0.5\n→ 0.606531\n\nУ первых двух графиков\nодинаковые оси. Далее\nмасштаб по вертикали меняется.",
                             "One x passes through\nthe whole chain.\n\nμ = 2, σ = 2, x = 4\n\n2 → 1 → 1 → −0.5\n→ 0.606531\n\nThe first two plots share\nthe same axes. Later plots\nuse different vertical scales."),
                       transform=axes.flat[-1].transAxes, va="top", color=INK, fontsize=12, linespacing=1.5)
    fig.suptitle(_text(language, "Из расстояния — в плавно убывающий вес",
                       "From distance to a smoothly decreasing weight"), fontsize=15, color=INK)
    return _finish(fig)


def width_figure(language="ru"):
    """Compare width changes and the convention represented by the factor 1/2."""
    coordinates = np.linspace(-6, 10, 801)
    mu = 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained", facecolor=PAPER)
    for ax in axes:
        _axes(ax)
        ax.set(xlim=(-6, 10), ylim=(0, 1.08), xlabel="x", ylabel="f(x)")
        ax.axvline(mu, color=MUTED, ls=":", alpha=0.4)
    axes[0].plot(coordinates, gaussian_values(coordinates, mu, 2), color=BLUE, lw=2.4, label="σ = 2")
    axes[0].plot(coordinates, gaussian_values(coordinates, mu, 1), color=ORANGE, lw=2.4, label="σ = 1")
    axes[0].set_title(_text(language, "Меняется ширина, максимум остаётся 1", "Width changes; the peak stays at 1"), fontsize=11, color=INK)
    axes[0].legend(frameon=False)
    z = (coordinates - mu) / 2
    axes[1].plot(coordinates, np.exp(-z ** 2 / 2), color=BLUE, lw=2.4, label="exp(−z² / 2)")
    axes[1].plot(coordinates, np.exp(-z ** 2), color=PURPLE, lw=2.4, label="exp(−z²)")
    axes[1].scatter([4, 4], [math.exp(-0.5), math.exp(-1)], color=[BLUE, PURPLE], s=45, zorder=5)
    axes[1].annotate("0.606531", (4, math.exp(-0.5)), xytext=(5.0, 0.67), color=BLUE, fontsize=9,
                     arrowprops=dict(arrowstyle="-", color=BLUE))
    axes[1].annotate("0.367879", (4, math.exp(-1)), xytext=(5.0, 0.37), color=PURPLE, fontsize=9,
                     arrowprops=dict(arrowstyle="-", color=PURPLE))
    axes[1].set_title(_text(language, "Убираем 1/2 при том же σ = 2", "Remove 1/2 at the same σ = 2"), fontsize=11, color=INK)
    axes[1].legend(loc="upper left", frameon=False)
    fig.suptitle(_text(language, "Масштаб и соглашение о ширине", "Scale and the width convention"), fontsize=15, color=INK)
    return _finish(fig)


@lru_cache(maxsize=1)
def _scene_arrays():
    count = 128
    indices = np.arange(count)
    vertical = 1 - 2 * (indices + 0.5) / count
    radial = np.sqrt(1 - vertical ** 2)
    angle = indices * math.pi * (3 - math.sqrt(5))
    normals = np.column_stack([radial * np.cos(angle), radial * np.sin(angle), vertical])
    centers = normals * 0.8 + np.array([0.0, 0.0, 1.1])
    covariances = []
    for normal in normals:
        tangent = np.cross(normal, [0.0, 0.0, 1.0])
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)
        rotation = np.column_stack([tangent, bitangent, normal])
        covariances.append(rotation @ np.diag(np.array([0.16, 0.13, 0.045]) ** 2) @ rotation.T)
    colors = np.column_stack([0.35 + 0.55 * (normals[:, 0] + 1) / 2,
                              0.30 + 0.50 * (normals[:, 1] + 1) / 2,
                              0.35 + 0.55 * (normals[:, 2] + 1) / 2])
    lin = np.linspace(-1.55, 1.55, 8)
    plane = np.array([[x, y, 0.0] for y in lin for x in lin])
    plane_colors = np.array([[0.24, 0.49, 0.57] if (i // 8 + i % 8) % 2 else [0.72, 0.82, 0.78]
                             for i in range(64)])
    means = np.vstack([centers, plane])
    covs = np.concatenate([np.array(covariances), np.repeat(np.diag([0.23 ** 2, 0.23 ** 2, 0.025 ** 2])[None], 64, axis=0)])
    rgb = np.vstack([colors, plane_colors])
    opacity = np.concatenate([np.full(128, 0.85), np.full(64, 0.88)])
    return means, covs, rgb, opacity


def scene_data():
    """Return independent arrays for a manually constructed 192-Gaussian scene.

    ``means`` is (192, 3), ``covariances`` is (192, 3, 3), ``colors`` is
    (192, 3) in [0, 1], and ``opacities`` is (192,). No training is involved.
    """
    return dict(zip(("means", "covariances", "colors", "opacities"),
                    (array.copy() for array in _scene_arrays())))


def project_scene(data=None, azimuth=-55.0, elevation=24.0):
    """Orthographically project 3D means and covariances; larger depth is nearer."""
    if data is None:
        data = scene_data()
    az, el = np.deg2rad([azimuth, elevation])
    toward_camera = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    up = np.cross(toward_camera, right)
    basis = np.stack([right, up])
    relative = np.asarray(data["means"]) - np.array([0.0, 0.0, 0.75])
    means = relative @ basis.T
    covariance = np.einsum("ai,nij,bj->nab", basis, data["covariances"], basis)
    depth = relative @ toward_camera
    return means, covariance, depth


def render_scene(data=None, azimuth=-55.0, elevation=24.0, zoom=1.0,
                 selected=0, opacity=None, isolate=False, size=260):
    """Return an RGB image using orthographic splats and center-depth blending.

    This is a teaching renderer, not the perspective 3DGS rasterizer. An elliptical
    Mahalanobis-radius-4 cutoff limits work; it is a numerical truncation, not a
    physical Gaussian boundary. The cutoff's omitted weight is exp(-8).
    """
    if data is None:
        data = scene_data()
    if not math.isfinite(zoom) or zoom <= 0:
        raise ValueError("zoom must be positive")
    if size < 8:
        raise ValueError("size must be at least 8")
    if not 0 <= selected < len(data["means"]):
        raise ValueError("selected index is outside the scene")
    if opacity is not None and not 0 <= opacity <= 1:
        raise ValueError("opacity must be in [0, 1]")
    means, covs, depths = project_scene(data, azimuth, elevation)
    half = 2.2 / zoom
    coordinates = np.linspace(-half, half, size)
    image = np.empty((size, size, 3), dtype=float)
    image[:] = np.array([244, 247, 251]) / 255
    order = [selected] if isolate else np.argsort(depths, kind="stable")
    for index in order:
        alpha = opacity if index == selected and opacity is not None else data["opacities"][index]
        if alpha == 0:
            continue
        center, cov = means[index], covs[index]
        radius = 4 * np.sqrt(np.diag(cov))
        low = np.searchsorted(coordinates, center - radius, side="left")
        high = np.searchsorted(coordinates, center + radius, side="right")
        x0, y0 = np.clip(low, 0, size)
        x1, y1 = np.clip(high, 0, size)
        if x0 >= x1 or y0 >= y1:
            continue
        xx, yy = np.meshgrid(coordinates[x0:x1] - center[0], coordinates[y0:y1] - center[1])
        inverse = np.linalg.inv(cov)
        distance2 = inverse[0, 0] * xx ** 2 + 2 * inverse[0, 1] * xx * yy + inverse[1, 1] * yy ** 2
        coverage = alpha * np.exp(-0.5 * distance2)
        coverage[distance2 > 16] = 0
        patch = image[y0:y1, x0:x1]
        patch[:] = coverage[..., None] * data["colors"][index] + (1 - coverage[..., None]) * patch
    return image


def _draw_scene(ax, mode, data, azimuth, elevation, zoom, selected=None, opacity=None, isolate=False):
    half = 2.2 / zoom
    means, covs, depth = project_scene(data, azimuth, elevation)
    ids = [selected] if isolate and selected is not None else np.argsort(depth, kind="stable")
    ax.set_facecolor(PAPER)
    if mode == "composite":
        rgb = render_scene(data, azimuth, elevation, zoom, selected or 0, opacity, isolate)
        ax.imshow(rgb, origin="lower", extent=[-half, half, -half, half], interpolation="nearest")
    elif mode == "centers":
        ids = np.asarray(ids)
        ax.scatter(means[ids, 0], means[ids, 1], c=data["colors"][ids], s=12, edgecolor="none")
    elif mode == "contours":
        for index in ids:
            eigenvalues, eigenvectors = np.linalg.eigh(covs[index])
            major = eigenvectors[:, 1]
            ellipse = Ellipse(means[index], 2 * np.sqrt(eigenvalues[1]), 2 * np.sqrt(eigenvalues[0]),
                              angle=np.rad2deg(np.arctan2(major[1], major[0])), facecolor="none",
                              edgecolor=data["colors"][index], lw=0.85, alpha=0.85)
            ax.add_patch(ellipse)
    else:
        raise ValueError("mode must be centers, contours or composite")
    if selected is not None:
        ax.scatter(*means[selected], s=95, marker="+", color=INK, linewidths=1.4, zorder=10)
    ax.set(xlim=(-half, half), ylim=(-half, half), aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d6e0ea")


def scene_overview(language="ru"):
    """Show the same hand-built scene and camera in three rendering modes."""
    data = scene_data()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), layout="constrained", facecolor="white")
    titles = _text(language, ["Центры", "Контуры веса 0.607", "Итоговое изображение"],
                   ["Centers", "Weight-0.607 contours", "Composite image"])
    for ax, mode, title in zip(axes, ("centers", "contours", "composite"), titles):
        _draw_scene(ax, mode, data, -55, 24, 1)
        ax.set_title(title, fontsize=12, color=INK, pad=10)
    fig.suptitle(_text(language, "Одна сцена · 192 гауссианы · одна камера",
                       "One scene · 192 Gaussians · one camera"), fontsize=16, color=INK)
    fig.supxlabel(_text(language, "Гауссианы заданы вручную. Ортографическая учебная проекция; обучение не выполняется.",
                         "Hand-built Gaussians. Orthographic teaching projection; no training is performed."), fontsize=10, color=MUTED)
    return _finish(fig)


def scene_figure(language="ru", azimuth=-55, elevation=24, zoom=1.0, mode="composite",
                 selected=0, opacity=0.85, isolate=False):
    """Single scene view with a plain-language card for the selected Gaussian."""
    data = scene_data()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.7), gridspec_kw={"width_ratios": [1.2, 1]},
                             layout="constrained", facecolor="white")
    _draw_scene(axes[0], mode, data, azimuth, elevation, zoom, selected, opacity, isolate)
    modes = _text(language, {"centers": "Центры", "contours": "Контуры веса 0.607", "composite": "Итоговое изображение"},
                  {"centers": "Centers", "contours": "Weight-0.607 contours", "composite": "Composite image"})
    axes[0].set_title(modes[mode], fontsize=13, color=INK)
    card = axes[1]
    card.axis("off")
    position = ", ".join(f"{v:.2f}" for v in data["means"][selected])
    scales = np.sqrt(np.linalg.eigvalsh(data["covariances"][selected]))[::-1]
    widths = ", ".join(f"{v:.3f}" for v in scales)
    rgb = ", ".join(f"{v:.2f}" for v in data["colors"][selected])
    component = _text(language, "Шар" if selected < 128 else "Плоскость", "Sphere" if selected < 128 else "Plane")
    card.text(0.03, 0.96, _text(language, f"Гауссиана № {selected}", f"Gaussian # {selected}"),
              fontsize=15, color=INK, va="top", weight="bold")
    orientation_ru = "Вдоль поверхности шара" if selected < 128 else "Параллельно плоскости"
    orientation_en = "Along the sphere's surface" if selected < 128 else "Parallel to the plane"
    body = _text(language,
                  f"{component}\n\nПоложение (x, y, z)\n({position})\n\nМасштабы вдоль собственных осей\n({widths})\n\nОриентация: {orientation_ru}\n\nЦвет (красный, зелёный, синий)\n({rgb})\n\nНепрозрачность в центре: {opacity:.2f}\nК краям вклад плавно уменьшается.",
                  f"{component}\n\nPosition (x, y, z)\n({position})\n\nScales along its own axes\n({widths})\n\nOrientation: {orientation_en}\n\nColor (red, green, blue)\n({rgb})\n\nOpacity at the center: {opacity:.2f}\nThe contribution fades toward the edges.")
    card.text(0.03, 0.85, body, fontsize=10.5, color=INK, va="top", linespacing=1.4)
    fig.supxlabel(_text(language, "Учебная сцена, заданная вручную · контур не является твёрдой границей",
                         "Hand-built teaching scene · a contour is not a solid boundary"), fontsize=10, color=MUTED)
    return _finish(fig)


def coverage_figure(language="ru"):
    """Top-down visibility diagram: hidden in one view is not necessarily unseen."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.7), layout="constrained", facecolor="white")
    for ax, both in zip(axes, (False, True)):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Circle((0, 0), 1, facecolor="#e6eef5", edgecolor=MUTED, lw=1.5))
        if both:
            ax.add_patch(Wedge((0, 0), 1, 180, 270, facecolor="#fce4d5", edgecolor=ORANGE, hatch="///", alpha=0.8))
        ax.plot([2.05], [0], marker="s", ms=13, color=BLUE)
        ax.annotate("", (1.15, 0), (1.85, 0), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))
        ax.text(2.05, -0.28, _text(language, "Камера A", "Camera A"), color=BLUE, fontsize=10, ha="center")
        if both:
            ax.plot([0], [2.0], marker="s", ms=13, color=PURPLE)
            ax.annotate("", (0, 1.15), (0, 1.8), arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.5))
            ax.text(0.25, 2.0, _text(language, "Камера B", "Camera B"), color=PURPLE, fontsize=10, va="center")
        for theta, color, label in ((0, GREEN, "P"), (120, PURPLE, "Q"), (225, ORANGE, "R")):
            angle = np.deg2rad(theta)
            point = np.array([np.cos(angle), np.sin(angle)])
            ax.scatter(*point, s=65, c=color, edgecolor="white", zorder=5)
            ax.text(*(point * 1.22), label, fontsize=12, color=color, weight="bold", ha="center", va="center")
        ax.set(xlim=(-1.6, 2.7), ylim=(-2.15, 2.5))
        title = _text(language, "Только камера A" if not both else "Камеры A и B",
                      "Camera A only" if not both else "Cameras A and B")
        ax.set_title(title, fontsize=14, color=INK)
        caption = _text(language,
                        "P видна. Q скрыта за объектом.\nПо одному снимку о Q известно мало." if not both else
                        "Q скрыта от A, но видна из B.\nR не видна ни из A, ни из B.",
                        "P is visible. Q is hidden behind the object.\nOne image tells us little about Q." if not both else
                        "Q is hidden from A but visible from B.\nR is visible from neither A nor B.")
        ax.text(0.4, -1.55, caption, ha="center", va="top", fontsize=10, color=INK, linespacing=1.6)
    fig.suptitle(_text(language, "Скрыто в одном кадре ≠ не наблюдалось вообще",
                       "Hidden in one image ≠ never observed"), fontsize=15, color=INK)
    fig.supxlabel(_text(language, "Схема сверху: непрозрачный круглый объект. Это иллюстрация видимости, а не реконструкция.",
                         "Top-down diagram of an opaque round object. This illustrates visibility, not reconstruction."), fontsize=10, color=MUTED)
    return _finish(fig)


def pipeline_figure(language="ru"):
    """Separate camera recovery, the Gaussian training loop and later viewing."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.7),
                             gridspec_kw={"height_ratios": [3.1, 1.1]},
                             layout="constrained", facecolor="white")

    def node(ax, x, y, heading, body, color=BLUE, width=0.20, height=0.22):
        ax.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                    boxstyle="round,pad=0.008,rounding_size=0.016",
                                    facecolor=PAPER, edgecolor=color, linewidth=1.3))
        ax.text(x, y + height * 0.24, heading, ha="center", va="center", fontsize=11,
                color=color, weight="bold")
        ax.text(x, y - height * 0.17, body, ha="center", va="center", fontsize=10,
                color=INK, linespacing=1.4)

    def arrow(ax, start, end, color=MUTED):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", color=color, linewidth=1.6,
                                    shrinkA=2, shrinkB=2, mutation_scale=13))

    training, viewing = axes
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
    training.set_title(_text(language, "От фотографий до обученной сцены",
                              "From photographs to a trained scene"), loc="left", fontsize=16, color=INK, pad=14)
    headings = _text(language, ["1. Фотографии", "2. Камеры и точки", "3. Гауссианы", "4. Рендер"],
                      ["1. Photographs", "2. Cameras + points", "3. Gaussians", "4. Render"])
    bodies = _text(language, ["Перекрывающиеся\nракурсы", "Оценка положений\nи начальной структуры", "Положение, форма,\nцвет, непрозрачность", "Изображение\nиз обучающей камеры"],
                    ["Overlapping\nviewpoints", "Estimate camera poses\nand initial structure", "Position, shape,\ncolor, opacity", "Image from\na training camera"])
    centers = [0.12, 0.37, 0.62, 0.87]
    for x, title, body in zip(centers, headings, bodies):
        node(training, x, 0.77, title, body)
    for first, second in zip(centers[:-1], centers[1:]):
        arrow(training, (first + 0.11, 0.77), (second - 0.11, 0.77))
    node(training, 0.87, 0.37, _text(language, "5. Сравнение", "5. Compare"),
         _text(language, "Насколько рендер\nотличается от снимка?", "How different is the\nrender from the photo?"), color=ORANGE)
    node(training, 0.62, 0.37, _text(language, "6. Обновление", "6. Update"),
         _text(language, "Меняем параметры\nгауссиан", "Change the Gaussian\nparameters"), color=GREEN)
    arrow(training, (0.87, 0.65), (0.87, 0.49), ORANGE)
    arrow(training, (0.76, 0.37), (0.73, 0.37), GREEN)
    arrow(training, (0.62, 0.49), (0.62, 0.65), GREEN)
    training.text(0.65, 0.565, _text(language, "повторяем", "repeat"), fontsize=9, color=GREEN,
                  va="center", ha="left")
    # The reference image follows a separate path, so it cannot be mistaken for
    # an output of the render or the parameter update.
    training.plot([0.12, 0.12, 0.87], [0.65, 0.13, 0.13], color=BLUE, lw=1.4)
    arrow(training, (0.87, 0.13), (0.87, 0.25), BLUE)
    training.text(0.15, 0.16, _text(language, "Исходный снимок для сравнения", "Original photograph for comparison"),
                  fontsize=10, color=BLUE, ha="left", va="bottom")
    training.text(0.12, 0.49, _text(language, "тот же\nракурс", "same\nviewpoint"),
                  fontsize=9, color=BLUE, ha="right", va="center",
                  bbox=dict(facecolor="white", edgecolor="none", pad=2))
    viewing.set_title(_text(language, "После обучения: просмотр с выбранного ракурса",
                             "After training: viewing from a chosen viewpoint"),
                      loc="left", fontsize=13, color=INK, pad=10)
    labels = _text(language,
                    [("Сохранённая модель", "Параметры гауссиан"), ("Новая камера", "Выбираем ракурс"), ("Новое изображение", "Рендерим модель")],
                    [("Saved model", "Gaussian parameters"), ("New camera", "Choose a viewpoint"), ("New image", "Render the model")])
    for x, (heading, body) in zip([0.16, 0.50, 0.84], labels):
        node(viewing, x, 0.55, heading, body, color=PURPLE, width=0.27, height=0.65)
    arrow(viewing, (0.305, 0.55), (0.355, 0.55), PURPLE)
    arrow(viewing, (0.645, 0.55), (0.695, 0.55), PURPLE)
    return _finish(fig)


def _widget_imports():
    import ipywidgets as widgets
    from IPython.display import clear_output, display
    return widgets, clear_output, display


def gaussian_lab(language="ru"):
    """Jupyter controls; ``course_controls`` exposes mu, sigma, x, preset and reset."""
    widgets, clear_output, display = _widget_imports()
    slider_layout = widgets.Layout(width="440px", max_width="100%")
    style = {"description_width": "120px"}
    controls = {
        "mu": widgets.FloatSlider(value=2, min=-3, max=3, step=0.1,
                                  description=_text(language, "Центр μ", "Center μ"), continuous_update=False, style=style, layout=slider_layout),
        "sigma": widgets.FloatSlider(value=2, min=0.25, max=3, step=0.05,
                                     description=_text(language, "Масштаб σ", "Scale σ"), continuous_update=False, style=style, layout=slider_layout),
        "x": widgets.FloatSlider(value=4, min=-12, max=12, step=0.1,
                                 description=_text(language, "Точка x", "Point x"), continuous_update=False, style=style, layout=slider_layout),
    }
    options = _text(language,
                    [("Исходный пример", "default"), ("Симметричная точка", "mirror"), ("Более узкий колокол", "narrow"), ("Точка в центре", "center"), ("Расстояние 3σ", "tail")],
                    [("Starting example", "default"), ("Symmetric point", "mirror"), ("Narrower bell", "narrow"), ("Point at the center", "center"), ("Distance 3σ", "tail")])
    controls["preset"] = widgets.Dropdown(options=options, value="default", description=_text(language, "Пример", "Example"), style=style, layout=slider_layout)
    controls["reset"] = widgets.Button(description=_text(language, "Сбросить", "Reset"), icon="undo")
    output = widgets.Output()
    busy = [False]

    def redraw(change=None):
        if busy[0]:
            return
        with output:
            clear_output(wait=True)
            display(gaussian_figure(controls["mu"].value, controls["sigma"].value, controls["x"].value, language))

    def set_values(mu, sigma, x):
        busy[0] = True
        controls["mu"].value, controls["sigma"].value, controls["x"].value = mu, sigma, x
        busy[0] = False
        redraw()

    presets = {"default": (2, 2, 4), "mirror": (2, 2, 0), "narrow": (2, 1, 4), "center": (2, 2, 2), "tail": (2, 2, 8)}
    for name in ("mu", "sigma", "x"):
        controls[name].observe(redraw, names="value")
    controls["preset"].observe(lambda change: set_values(*presets[change["new"]]), names="value")

    def reset(button):
        busy[0] = True
        controls["preset"].value = "default"
        busy[0] = False
        set_values(2, 2, 4)

    controls["reset"].on_click(reset)
    hint = widgets.HTML(value=_text(language,
                                    "<b>Измени один параметр.</b> Значение обновляется после отпускания ползунка. Оси графика остаются на месте.",
                                    "<b>Change one parameter.</b> Values update when you release the slider. The plot axes stay fixed."))
    box = widgets.VBox([hint, controls["mu"], controls["sigma"], controls["x"], controls["preset"], controls["reset"], output])
    box.course_controls = controls
    box.course_output = output
    box.course_redraw = redraw
    redraw()
    return box


def scene_lab(language="ru"):
    """Jupyter scene explorer, fully local and independent of a GPU or dataset."""
    widgets, clear_output, display = _widget_imports()
    slider_layout = widgets.Layout(width="440px", max_width="100%")
    style = {"description_width": "140px"}

    def slider(value, minimum, maximum, step, ru, en):
        return widgets.FloatSlider(value=value, min=minimum, max=maximum, step=step,
                                   description=_text(language, ru, en), continuous_update=False,
                                   style=style, layout=slider_layout)

    controls = {
        "azimuth": slider(-55, -180, 180, 5, "Поворот камеры, °", "Camera orbit, °"),
        "elevation": slider(24, -10, 80, 2, "Высота взгляда, °", "View elevation, °"),
        "zoom": slider(1, 0.65, 1.8, 0.05, "Приближение", "Zoom"),
        "selected": widgets.IntSlider(value=0, min=0, max=191, step=1,
                                       description=_text(language, "Номер гауссианы", "Gaussian index"), continuous_update=False, style=style, layout=slider_layout),
        "opacity": slider(0.85, 0, 1, 0.05, "Непрозрачность", "Opacity"),
        "isolate": widgets.Checkbox(value=False, description=_text(language, "Только выбранная гауссиана", "Only the selected Gaussian"), indent=False),
    }
    options = _text(language, [("Центры", "centers"), ("Контуры", "contours"), ("Итоговое изображение", "composite")],
                    [("Centers", "centers"), ("Contours", "contours"), ("Composite image", "composite")])
    controls["mode"] = widgets.Dropdown(options=options, value="composite", description=_text(language, "Режим", "Mode"), style=style, layout=slider_layout)
    controls["reset"] = widgets.Button(description=_text(language, "Сбросить", "Reset"), icon="undo")
    output = widgets.Output()
    busy = [False]

    def redraw(change=None):
        if busy[0]:
            return
        with output:
            clear_output(wait=True)
            values = {name: controls[name].value for name in ("azimuth", "elevation", "zoom", "mode", "selected", "opacity", "isolate")}
            display(scene_figure(language=language, **values))

    def select(change):
        busy[0] = True
        controls["opacity"].value = float(scene_data()["opacities"][change["new"]])
        busy[0] = False
        redraw()

    for name in ("azimuth", "elevation", "zoom", "opacity", "mode", "isolate"):
        controls[name].observe(redraw, names="value")
    controls["selected"].observe(select, names="value")

    def reset(button):
        busy[0] = True
        defaults = dict(azimuth=-55, elevation=24, zoom=1, selected=0, opacity=0.85, isolate=False, mode="composite")
        for name, value in defaults.items():
            controls[name].value = value
        busy[0] = False
        redraw()

    controls["reset"].on_click(reset)
    hint = widgets.HTML(value=_text(language,
                                    "<b>Учебная синтетическая сцена.</b> Гауссианы заданы вручную; обучения по фотографиям здесь нет. Непрозрачность меняет вклад в итоговую картинку, а не диагностические центры и контуры.",
                                    "<b>Synthetic teaching scene.</b> The Gaussians are hand-built; no training from photographs takes place. Opacity changes the composite contribution, not the diagnostic centers or contours."))
    box = widgets.VBox([hint, controls["mode"], controls["azimuth"], controls["elevation"], controls["zoom"],
                        controls["selected"], controls["opacity"], controls["isolate"], controls["reset"], output])
    box.course_controls = controls
    box.course_output = output
    box.course_redraw = redraw
    redraw()
    return box
