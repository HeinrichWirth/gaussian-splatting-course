"""Three views, slices, and Jupyter controls for a Gaussian on a plane."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from gaussian_math import gaussian2d, gaussian2d_steps


BLUE = "#2463a9"
ORANGE = "#d85b27"
GREEN = "#23866c"
INK = "#26364a"
MUTED = "#52647a"
PAPER = "#f4f7fb"
LIMIT = 5.0


def _text(language, ru, en):
    if language not in ("ru", "en"):
        raise ValueError("language must be 'ru' or 'en'")
    return ru if language == "ru" else en


def _finish(fig):
    plt.close(fig)
    return fig


def _axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#b6c3d1")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(alpha=.16)
    ax.set_axisbelow(True)


def _plane(ax):
    ax.set(xlim=(-LIMIT, LIMIT), ylim=(-LIMIT, LIMIT), xlabel="x", ylabel="y")
    ax.set_aspect("equal")
    ax.tick_params(labelsize=9)


def _number(value):
    return f"{value:.3e}" if 0 < abs(value) < 1e-5 else f"{value:.3f}".rstrip("0").rstrip(".")


def _mesh(points=161):
    coordinates = np.linspace(-LIMIT, LIMIT, points)
    return np.meshgrid(coordinates, coordinates)


def plane_figure(mux=0, muy=0, sx=2, sy=1, x=2, y=1, language="ru"):
    """Show the same fixed-domain function as a heatmap, a hill and contours."""
    steps = gaussian2d_steps(x, y, mux, muy, sx, sy)
    xx, yy = _mesh()
    zz = gaussian2d(xx, yy, mux, muy, sx, sy)
    levels = [math.exp(-4.5), math.exp(-2), math.exp(-.5)]
    fig = plt.figure(figsize=(12.4, 5.5), layout="constrained", facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[3.5, 1.0])
    heat = fig.add_subplot(grid[0, 0])
    hill = fig.add_subplot(grid[0, 1], projection="3d", computed_zorder=False)
    contours = fig.add_subplot(grid[0, 2])
    table_ax = fig.add_subplot(grid[1, :])

    colored = heat.imshow(zz, origin="lower", extent=(-LIMIT, LIMIT, -LIMIT, LIMIT),
                          cmap="magma", vmin=0, vmax=1, interpolation="bilinear")
    colorbar = fig.colorbar(colored, ax=heat, orientation="horizontal", fraction=.045, pad=.13,
                            ticks=[0, .5, 1], shrink=.85)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label(_text(language, "Вес", "Weight"), fontsize=9)
    heat.scatter([x], [y], marker="x", s=65, color="cyan", linewidth=2, zorder=4)
    heat.scatter([mux], [muy], marker="+", s=85, color="white", linewidth=1.3)
    _plane(heat)
    heat.set_title(_text(language, "Вес показан цветом", "Weight shown as color"), color=INK, fontsize=12)

    # Include the center and the probe explicitly: a coarse drawing grid should
    # not make the true peak or the sampled point appear lower than its value.
    hill_x = np.unique(np.r_[np.linspace(-LIMIT, LIMIT, 61), mux, x])
    hill_y = np.unique(np.r_[np.linspace(-LIMIT, LIMIT, 61), muy, y])
    hx, hy = np.meshgrid(hill_x, hill_y)
    hz = gaussian2d(hx, hy, mux, muy, sx, sy)
    hill.plot_surface(hx, hy, hz, cmap="magma", vmin=0, vmax=1,
                      rcount=len(hill_y), ccount=len(hill_x), linewidth=0,
                      antialiased=True, alpha=.94, zorder=0)
    hill.plot([x, x], [y, y], [0, steps["weight"]], color=ORANGE, lw=1.5, zorder=5)
    hill.scatter([x], [y], [steps["weight"]], color=ORANGE, marker="x", s=55,
                  linewidth=2, depthshade=False, zorder=6)
    hill.set(xlim=(-LIMIT, LIMIT), ylim=(-LIMIT, LIMIT), zlim=(0, 1.05),
             xlabel="x", ylabel="y", zlabel=_text(language, "Вес", "Weight"))
    hill.set_box_aspect((1, 1, .75))
    hill.view_init(elev=28, azim=-58)
    hill.tick_params(labelsize=8, pad=1)
    hill.set_title(_text(language, "Вес показан высотой", "Weight shown as height"), color=INK, fontsize=12)

    cs = contours.contour(xx, yy, zz, levels=levels, colors=["#abc2de", BLUE, GREEN], linewidths=[1, 1.4, 1.8])
    contours.clabel(cs, fmt={value: f"{value:.3f}" for value in levels}, fontsize=8)
    contours.scatter([x], [y], marker="x", s=65, color=ORANGE, linewidth=2, zorder=4)
    contours.scatter([mux], [muy], marker="+", s=85, color=INK, linewidth=1.3)
    _plane(contours)
    contours.set_title(_text(language, "На линии вес одинаков", "Equal weight along each line"), color=INK, fontsize=12)
    contours.grid(alpha=.16)

    fig.suptitle(_text(language, "Одна функция, три вида", "One function, three views"), fontsize=17, color=INK)
    table_ax.axis("off")
    titles = _text(language, ["1. Смещения", "2. Делим на ширины", "3. Складываем квадраты", "4. Получаем вес"],
                   ["1. Offsets", "2. Divide by the scales", "3. Add the squares", "4. Find the weight"])
    values = [f"dx = {_number(x)} − ({_number(mux)}) = {_number(steps['dx'])}\ndy = {_number(y)} − ({_number(muy)}) = {_number(steps['dy'])}",
              f"ux = {_number(steps['dx'])} / {_number(sx)} = {_number(steps['ux'])}\nuy = {_number(steps['dy'])} / {_number(sy)} = {_number(steps['uy'])}",
              f"q = ({_number(steps['ux'])})² + ({_number(steps['uy'])})²\nq = {_number(steps['q'])}",
              f"exp(−q / 2) = exp({_number(steps['exponent'])})\nf(x, y) = {steps['weight']:.6g}"]
    table = table_ax.table(cellText=[values], colLabels=titles, colWidths=[.25]*4,
                           cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.25)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dce4ee")
        cell.set_facecolor(PAPER if row == 0 else "white")
        cell.set_text_props(color=INK, weight="bold" if row == 0 else "normal")
    return _finish(fig)


def slices_figure(mux=0, muy=0, sx=2, sy=1, x=2, y=1, language="ru"):
    """Two cuts through the selected point, compared with cuts through the peak."""
    steps = gaussian2d_steps(x, y, mux, muy, sx, sy)
    coordinates = np.linspace(-LIMIT, LIMIT, 501)
    xx, yy = _mesh()
    weights = gaussian2d(xx, yy, mux, muy, sx, sy)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), layout="constrained", facecolor="white")
    axes[0].imshow(weights, origin="lower", extent=(-LIMIT, LIMIT, -LIMIT, LIMIT),
                   cmap="magma", vmin=0, vmax=1, interpolation="bilinear")
    axes[0].axhline(y, color="cyan", lw=1.6)
    axes[0].axvline(x, color="lime", lw=1.6)
    axes[0].scatter([x], [y], marker="x", color="white", s=70, zorder=4)
    _plane(axes[0])
    axes[0].set_title(_text(language, "Разрезаем пятно двумя линиями", "Two cuts through the spot"), fontsize=11, color=INK)

    cuts = [(gaussian2d(coordinates, y, mux, muy, sx, sy),
             gaussian2d(coordinates, muy, mux, muy, sx, sy), x, BLUE, "x", f"y = {_number(y)}"),
            (gaussian2d(x, coordinates, mux, muy, sx, sy),
             gaussian2d(mux, coordinates, mux, muy, sx, sy), y, GREEN, "y", f"x = {_number(x)}")]
    for ax, (cut, central, sample, color, axis, fixed) in zip(axes[1:], cuts):
        ax.plot(coordinates, central, "--", color="#a2adba", lw=1.5,
                 label=_text(language, "Срез через центр", "Cut through the center"))
        ax.plot(coordinates, cut, color=color, lw=2,
                 label=_text(language, "Выбранный срез", "Selected cut"))
        ax.scatter([sample], [steps["weight"]], color=ORANGE, s=38, zorder=5)
        ax.axvline(sample, color=ORANGE, lw=.8, alpha=.6)
        ax.set(xlim=(-LIMIT, LIMIT), ylim=(0, 1.05), xlabel=axis,
               ylabel=_text(language, "Вес", "Weight"))
        ax.set_title(_text(language, f"Двигаем {axis}; {fixed} закреплено", f"Vary {axis}; keep {fixed}"), fontsize=11, color=INK)
        _axes(ax)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(_text(language,
                       f"В месте пересечения оба среза дают {steps['weight']:.6f}",
                       f"Both cuts give {steps['weight']:.6f} at the intersection"), fontsize=14, color=INK)
    return _finish(fig)


def ellipse_figure(language="ru"):
    """A circle and an ellipse at the same weight, with three numeric probes."""
    xx, yy = _mesh()
    level = math.exp(-.5)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.4), layout="constrained", facecolor="white")
    settings = [(1, 1), (2, 1)]
    titles = _text(language, ["σx = σy = 1: круг", "σx = 2, σy = 1: эллипс"],
                   ["σx = σy = 1: circle", "σx = 2, σy = 1: ellipse"])
    for ax, (sx, sy), title in zip(axes[:2], settings, titles):
        ax.contour(xx, yy, gaussian2d(xx, yy, sx=sx, sy=sy), levels=[level], colors=[BLUE], linewidths=2)
        ax.axhline(0, color="#b6c3d1", lw=.8)
        ax.axvline(0, color="#b6c3d1", lw=.8)
        ax.plot([0, sx], [0, 0], color=ORANGE, lw=2)
        ax.plot([0, 0], [0, sy], color=GREEN, lw=2)
        ax.scatter([sx, 0], [0, sy], c=[ORANGE, GREEN], s=45, zorder=5)
        ax.text(sx / 2, -.35, "σx", color=ORANGE, ha="center", fontsize=11)
        ax.text(-.35, sy / 2, "σy", color=GREEN, ha="right", fontsize=11)
        _plane(ax)
        ax.set_title(title, fontsize=12, color=INK)
        ax.grid(alpha=.16)
    axes[2].axis("off")
    rows = [
        ["(2, 0)", "1² + 0² = 1", "0.606531"],
        ["(0, 1)", "0² + 1² = 1", "0.606531"],
        ["(√2, 1/√2)", "0.5 + 0.5 = 1", "0.606531"],
        ["(2, 1)", "1² + 1² = 2", "0.367879"],
    ]
    table = axes[2].table(cellText=rows,
                          colLabels=_text(language, ["Точка", "ux² + uy²", "Вес"], ["Point", "ux² + uy²", "Weight"]),
                          colWidths=[.34, .40, .30], cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.1)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dce4ee")
        cell.set_facecolor(PAPER if row == 0 else "white")
        cell.set_text_props(color=INK, weight="bold" if row == 0 else "normal")
    axes[2].set_title(_text(language, "Проверка точек эллипса", "Checking points on the ellipse"), color=INK, fontsize=12)
    fig.suptitle(_text(language, "На синей линии вес равен exp(−1/2) ≈ 0.606531",
                       "Along the blue line, the weight is exp(−1/2) ≈ 0.606531"), color=INK, fontsize=14)
    return _finish(fig)


def plane_lab(language="ru"):
    """Interactive axis-aligned Gaussian; all coordinates and colors use fixed scales."""
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    style = {"description_width": "145px"}
    layout = widgets.Layout(width="455px", max_width="100%")

    def slider(value, minimum, maximum, step, ru, en):
        return widgets.FloatSlider(value=value, min=minimum, max=maximum, step=step,
                                   description=_text(language, ru, en), continuous_update=False,
                                   style=style, layout=layout, readout_format=".2f")

    controls = {
        "mux": slider(0, -2, 2, .1, "Центр по x, μx", "Center x, μx"),
        "muy": slider(0, -2, 2, .1, "Центр по y, μy", "Center y, μy"),
        "sx": slider(2, .3, 2.5, .1, "Ширина по x, σx", "Scale x, σx"),
        "sy": slider(1, .3, 2.5, .1, "Ширина по y, σy", "Scale y, σy"),
        "x": slider(2, -4, 4, .1, "Точка: x", "Point: x"),
        "y": slider(1, -4, 4, .1, "Точка: y", "Point: y"),
    }
    controls["preset"] = widgets.Dropdown(
        options=_text(language,
                      [("Исходный пример", "default"), ("Круг", "circle"), ("Вытянуть по y", "vertical"),
                       ("Точка на одной ширине", "one_width"), ("Точка в центре", "center")],
                      [("Starting example", "default"), ("Circle", "circle"), ("Stretch along y", "vertical"),
                       ("Point one scale away", "one_width"), ("Point at the center", "center")]),
        value="default", description=_text(language, "Пример", "Example"), style=style, layout=layout)
    controls["view"] = widgets.Dropdown(options=_text(language, [("Три вида", "views"), ("Срезы", "slices")],
                                                     [("Three views", "views"), ("Slices", "slices")]),
                                        value="views", description=_text(language, "Показать", "Show"), style=style, layout=layout)
    controls["reset"] = widgets.Button(description=_text(language, "Сбросить", "Reset"), icon="undo")
    output = widgets.Output()
    busy = [False]
    names = ("mux", "muy", "sx", "sy", "x", "y")
    defaults = (0, 0, 2, 1, 2, 1)
    presets = {"default": defaults, "circle": (0, 0, 1, 1, 1, 0),
               "vertical": (0, 0, 1, 2, 1, 2), "one_width": (0, 0, 2, 1, 2, 0),
               "center": (0, 0, 2, 1, 0, 0)}

    def redraw(change=None):
        if busy[0]:
            return
        values = {name: controls[name].value for name in names}
        function = plane_figure if controls["view"].value == "views" else slices_figure
        with output:
            clear_output(wait=True)
            display(function(language=language, **values))

    def set_values(values):
        busy[0] = True
        for name, value in zip(names, values):
            controls[name].value = value
        busy[0] = False
        redraw()

    for name in (*names, "view"):
        controls[name].observe(redraw, names="value")
    controls["preset"].observe(
        lambda change: None if busy[0] else set_values(presets[change["new"]]), names="value")

    def reset(button):
        busy[0] = True
        controls["preset"].value = "default"
        controls["view"].value = "views"
        busy[0] = False
        set_values(defaults)

    controls["reset"].on_click(reset)
    hint = widgets.HTML(value=_text(language,
        "<b>Меняй по одному числу.</b> Крестик — выбранная точка. Пределы координат, высота 1 и шкала цвета остаются прежними. Значения обновляются после отпускания ползунка.",
        "<b>Change one number at a time.</b> The cross marks the selected point. Coordinate limits, peak height 1, and the color scale stay fixed. Values update when you release the slider."))
    box = widgets.VBox([hint, controls["view"], *[controls[name] for name in names],
                        controls["preset"], controls["reset"], output])
    box.course_controls = controls
    box.course_output = output
    box.course_redraw = redraw
    redraw()
    return box
