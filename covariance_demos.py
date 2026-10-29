"""Local, bilingual figures and controls for covariance and rotation in lesson 4."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle

from gaussian_math import covariance_matrix, gaussian2d, gaussian2d_steps, rotation_matrix


BLUE = "#2463a9"
ORANGE = "#d85b27"
GREEN = "#23866c"
PURPLE = "#8256a9"
INK = "#26364a"
PAPER = "#f4f7fb"


def _text(language, ru, en):
    if language not in ("ru", "en"):
        raise ValueError("language must be 'ru' or 'en'")
    return ru if language == "ru" else en


def _finish(fig):
    plt.close(fig)
    return fig


def _axes(ax, limit=4):
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
    ax.axhline(0, color="#9aaabb", lw=0.7)
    ax.axvline(0, color="#9aaabb", lw=0.7)
    ax.grid(alpha=0.17)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("x", color=INK)
    ax.set_ylabel("y", color=INK)
    ax.spines[["top", "right"]].set_visible(False)


def _fmt(value, digits=3):
    value = 0.0 if abs(value) < 0.5 * 10 ** (-digits) else value
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") or "0"


def _matrix_text(matrix):
    return "\n".join("[ " + "  ".join(f"{_fmt(v):>7}" for v in row) + " ]" for row in matrix)


def covariance_figure(s1=2, s2=1, angle_deg=45, x=1, y=1, language="ru"):
    """A fixed-view ellipse, its covariance, and a complete point calculation."""
    values = gaussian2d_steps(x=x, y=y, sx=s1, sy=s2, angle_deg=angle_deg)
    sigma = covariance_matrix(s1, s2, angle_deg)
    inverse = np.linalg.inv(sigma)
    coords = np.linspace(-4, 4, 241)
    xx, yy = np.meshgrid(coords, coords)
    weights = gaussian2d(xx, yy, sx=s1, sy=s2, angle_deg=angle_deg)
    rotation = rotation_matrix(angle_deg)
    parameter = np.linspace(0, 2 * np.pi, 361)
    contour = rotation @ np.vstack((s1 * np.cos(parameter), s2 * np.sin(parameter)))
    fig = plt.figure(figsize=(10.8, 5.4), layout="constrained", facecolor=PAPER)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.1, 1))
    ax = fig.add_subplot(gs[0, 0])
    _axes(ax)
    ax.imshow(weights, origin="lower", extent=(-4, 4, -4, 4), vmin=0, vmax=1,
              cmap="Blues", interpolation="bilinear")
    ax.plot(*contour, color=BLUE, lw=1.8, label=_text(language, "Контур веса 0.607", "Weight 0.607 contour"))
    for k, (scale, color) in enumerate(((s1, ORANGE), (s2, GREEN))):
        direction = rotation[:, k] * scale
        ax.annotate("", direction, (0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=2.4))
        ax.annotate(f"s{k + 1}", direction, xytext=(5, 5), textcoords="offset points", color=color, fontsize=11)
    ax.scatter([x], [y], s=55, facecolor=PURPLE, edgecolor="white", zorder=5)
    ax.annotate(f"({_fmt(x)}, {_fmt(y)})", (x, y), xytext=(7, -16), textcoords="offset points", fontsize=10, color=PURPLE)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.set_title(_text(language, "Пятно и его собственные оси", "The spot and its own axes"), fontsize=12, color=INK)
    card = fig.add_subplot(gs[0, 1])
    card.axis("off")
    card.text(0.02, 0.98, f"s1 = {_fmt(s1)},  s2 = {_fmt(s2)},  θ = {_fmt(angle_deg)}°",
              va="top", fontsize=12, weight="bold", color=INK)
    card.text(0.02, 0.87, "Σ =\n" + _matrix_text(sigma), va="top", fontsize=12, family="monospace", color=BLUE)
    card.text(0.02, 0.65, "Σ⁻¹ =\n" + _matrix_text(inverse), va="top", fontsize=12, family="monospace", color=GREEN)
    lines = [
        _text(language, "Поворот точки назад:", "Rotate the point back:"),
        f"d1 = {_fmt(values['local_x'], 6)},  d2 = {_fmt(values['local_y'], 6)}",
        f"u1 = d1 / s1 = {_fmt(values['ux'], 6)}",
        f"u2 = d2 / s2 = {_fmt(values['uy'], 6)}",
        f"q = u1² + u2² = {_fmt(values['q'], 6)}",
        f"f = exp(−q / 2) = {values['weight']:.6g}",
    ]
    card.text(0.02, 0.41, "\n".join(lines), va="top", fontsize=10.6, linespacing=1.65, color=INK)
    fig.suptitle(_text(language, "Повернули эллипс — изменились числа в матрице", "Rotate the ellipse and watch the matrix change"), fontsize=14, color=INK)
    return _finish(fig)


def rotation_figure(angle_deg=45, language="ru"):
    """Show sine/cosine as signed coordinates, then the two rotation columns."""
    rotation = rotation_matrix(angle_deg)
    c, s = rotation[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.1), layout="constrained", facecolor=PAPER)
    ax = axes[0]
    _axes(ax, 1.4)
    ax.add_patch(Circle((0, 0), 1, fill=False, color="#b6c3d1", lw=1.4))
    ax.plot([c, c], [0, s], "--", color=GREEN, lw=1.8)
    ax.plot([0, c], [0, 0], color=ORANGE, lw=3)
    ax.annotate("", (c, s), (0, 0), arrowprops=dict(arrowstyle="->", color=BLUE, lw=2.6))
    ax.scatter([c], [s], color=BLUE, s=35)
    ax.annotate(f"(cos θ, sin θ)\n({_fmt(c)}, {_fmt(s)})", (c, s), xytext=(7, 10),
                textcoords="offset points", color=BLUE, fontsize=10)
    ax.text(c / 2, -0.12 if s >= 0 else 0.1, f"cos θ = {_fmt(c)}", ha="center", color=ORANGE, fontsize=10)
    ax.text(c + 0.1, s / 2, f"sin θ\n= {_fmt(s)}", va="center", color=GREEN, fontsize=10)
    if 0 <= angle_deg <= 180:
        ax.add_patch(Arc((0, 0), 0.6, 0.6, theta1=0, theta2=angle_deg, color=BLUE))
        ax.text(0.35, 0.14, "θ", color=BLUE, fontsize=11)
    ax.set_title(_text(language, "Стрелка длины 1", "An arrow of length 1"), color=INK, fontsize=12)
    ax = axes[1]
    _axes(ax, 1.4)
    for k, color in ((0, ORANGE), (1, GREEN)):
        ax.annotate("", rotation[:, k], (0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=2.7))
        label = "(cos θ, sin θ)" if k == 0 else "(−sin θ, cos θ)"
        ax.annotate(label, rotation[:, k], xytext=(4, 8), textcoords="offset points", color=color, fontsize=10)
    ax.text(0, -1.05, "R = [[cos θ, −sin θ],\n       [sin θ,  cos θ]]", ha="center", va="center", fontsize=10.5, family="monospace", color=INK)
    ax.set_title(_text(language, "Два столбца R — две новые оси", "The two columns of R are the new axes"), color=INK, fontsize=12)
    fig.suptitle(_text(language, f"Поворот против часовой стрелки: {angle_deg:g}°", f"Counterclockwise rotation: {angle_deg:g}°"), fontsize=14, color=INK)
    return _finish(fig)


def sample_covariance_figure(angle_deg=45, language="ru"):
    """Four symmetric observations; averages use division by N=4 explicitly."""
    original = np.array([[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]])
    rotated = original @ rotation_matrix(angle_deg).T
    fig = plt.figure(figsize=(10.8, 7.0), layout="constrained", facecolor=PAPER)
    gs = fig.add_gridspec(2, 2, height_ratios=(1.7, 1))
    for j, points in enumerate((original, rotated)):
        ax = fig.add_subplot(gs[0, j])
        _axes(ax, 3.3)
        ax.scatter(points[:, 0], points[:, 1], color=BLUE if j == 0 else PURPLE, s=50)
        for label, point in zip("ABCD", points):
            ax.annotate(label, point, xytext=(6, 6), textcoords="offset points", fontsize=10)
        rectangle = points[[0, 1, 3, 2, 0]]
        ax.plot(rectangle[:, 0], rectangle[:, 1], color="#9aaabb", lw=1)
        title = _text(language, "До поворота: средние x и y равны 0", "Before rotation: mean x and mean y are 0") if j == 0 else _text(language, f"После поворота на {angle_deg:g}°", f"After a {angle_deg:g}° rotation")
        ax.set_title(title, fontsize=11, color=INK)
    table_ax = fig.add_subplot(gs[1, :])
    table_ax.axis("off")
    rows = []
    for label, (x, y) in zip("ABCD", rotated):
        rows.append([label, _fmt(x), _fmt(y), _fmt(x*x), _fmt(x*y), _fmt(y*y)])
    products = np.column_stack((rotated[:, 0] ** 2, rotated[:, 0] * rotated[:, 1], rotated[:, 1] ** 2))
    rows.append([_text(language, "Среднее", "Mean"), "0", "0", *[_fmt(v) for v in products.mean(axis=0)]])
    table = table_ax.table(cellText=rows, colLabels=[_text(language, "Точка", "Point"), "x", "y", "x²", "x·y", "y²"],
                           loc="center", cellLoc="center", colColours=["#e7eef6"] * 6)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d7e0e9")
        if row == 5:
            cell.set_facecolor("#eaf4ef")
    fig.suptitle(_text(language, "Ковариация на четырёх точках: складываем произведения и делим на 4", "Covariance from four points: add the products and divide by 4"), fontsize=13, color=INK)
    return _finish(fig)


def inverse_figure(language="ru"):
    """The same q=1 contour and sample point in three coordinate systems."""
    angle = np.linspace(0, 2 * np.pi, 361)
    local_contour = np.vstack((2 * np.cos(angle), np.sin(angle)))
    world_contour = rotation_matrix(45) @ local_contour
    normalized = np.vstack((np.cos(angle), np.sin(angle)))
    contours = (world_contour, local_contour, normalized)
    points = ((1., 1.), (math.sqrt(2), 0.), (1/math.sqrt(2), 0.))
    titles = _text(language, ["1. Исходный эллипс", "2. Повернули точку назад", "3. Разделили на s1 и s2"],
                   ["1. Original ellipse", "2. Rotate the point back", "3. Divide by s1 and s2"])
    labels = _text(language, ["d = (1, 1)", "d_local ≈ (1.414, 0)", "u ≈ (0.707, 0)"],
                   ["d = (1, 1)", "d_local ≈ (1.414, 0)", "u ≈ (0.707, 0)"])
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.4), layout="constrained", facecolor=PAPER)
    for ax, contour, point, title, label in zip(axes, contours, points, titles, labels):
        _axes(ax, 2.6)
        ax.fill(*contour, color=BLUE, alpha=0.08)
        ax.plot(*contour, color=BLUE, lw=2)
        ax.scatter(*point, color=PURPLE, s=45, zorder=4)
        ax.plot([0, point[0]], [0, point[1]], color=PURPLE, lw=1.5)
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.text(0.5, -0.20, label, transform=ax.transAxes, ha="center", fontsize=10, color=INK)
    axes[1].set(xlabel="d1", ylabel="d2")
    axes[2].set(xlabel="u1", ylabel="u2")
    fig.suptitle(_text(language, "Один расчёт: q = (1/√2)² + 0² = 0.5,  вес ≈ 0.778801", "One calculation: q = (1/√2)² + 0² = 0.5,  weight ≈ 0.778801"), fontsize=13, color=INK)
    return _finish(fig)


def covariance_lab(language="ru"):
    """Rotate an ellipse, inspect covariance entries and calculate a chosen point."""
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    layout = widgets.Layout(width="460px", max_width="100%")
    style = {"description_width": "150px"}

    def slider(value, minimum, maximum, step, ru, en):
        return widgets.FloatSlider(value=value, min=minimum, max=maximum, step=step,
                                   description=_text(language, ru, en), continuous_update=False,
                                   layout=layout, style=style, readout_format=".2f")

    controls = {
        "s1": slider(2, 0.25, 3, 0.05, "Масштаб оси s1", "Axis scale s1"),
        "s2": slider(1, 0.25, 3, 0.05, "Масштаб оси s2", "Axis scale s2"),
        "angle": slider(45, -180, 180, 5, "Поворот θ, °", "Rotation θ, °"),
        "x": slider(1, -4, 4, 0.25, "Точка x", "Point x"),
        "y": slider(1, -4, 4, 0.25, "Точка y", "Point y"),
    }
    presets = {"default": (2, 1, 45, 1, 1), "axis": (2, 1, 0, 1, 1),
               "opposite": (2, 1, -45, 1, 1), "circle": (1.5, 1.5, 45, 1, 1),
               "center": (2, 1, 45, 0, 0)}
    options = _text(language,
                    [("Основной пример", "default"), ("Оси совпадают с x и y", "axis"), ("Наклон в другую сторону", "opposite"), ("Круг", "circle"), ("Точка в центре", "center")],
                    [("Main example", "default"), ("Axes aligned with x and y", "axis"), ("Tilt the other way", "opposite"), ("Circle", "circle"), ("Point at the center", "center")])
    controls["preset"] = widgets.Dropdown(options=options, value="default", description=_text(language, "Пример", "Example"), layout=layout, style=style)
    controls["reset"] = widgets.Button(description=_text(language, "Сбросить", "Reset"), icon="undo")
    output = widgets.Output()
    busy = [False]

    def redraw(change=None):
        if busy[0]:
            return
        with output:
            clear_output(wait=True)
            display(covariance_figure(controls["s1"].value, controls["s2"].value, controls["angle"].value,
                                      controls["x"].value, controls["y"].value, language=language))

    def set_values(name):
        busy[0] = True
        for key, value in zip(("s1", "s2", "angle", "x", "y"), presets[name]):
            controls[key].value = value
        busy[0] = False
        redraw()

    for key in ("s1", "s2", "angle", "x", "y"):
        controls[key].observe(redraw, names="value")
    controls["preset"].observe(lambda change: set_values(change["new"]) if not busy[0] else None, names="value")

    def reset(button):
        busy[0] = True
        controls["preset"].value = "default"
        busy[0] = False
        set_values("default")

    controls["reset"].on_click(reset)
    hint = widgets.HTML(value=_text(language,
        "<b>Поверни пятно и посмотри на матрицу.</b> s1 и s2 — масштабы вдоль его собственных осей. Центр здесь всегда (0, 0). Ползунки обновляют рисунок после отпускания.",
        "<b>Rotate the spot and watch the matrix.</b> s1 and s2 are scales along its own axes. The center stays at (0, 0). The plot updates when you release a slider."))
    box = widgets.VBox([hint, *[controls[key] for key in ("s1", "s2", "angle", "x", "y", "preset", "reset")], output])
    box.course_controls = controls
    box.course_output = output
    box.course_redraw = redraw
    redraw()
    return box
