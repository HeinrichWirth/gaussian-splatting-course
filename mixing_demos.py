"""Color compositing experiments and a small, offline 2D Gaussian editor."""

from __future__ import annotations

import base64
import copy
import html

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle

from gaussian_math import (
    composite_at, gaussian2d, load_scene_json, make_layer, render_layers, scene_json,
)


INK = "#26364a"
MUTED = "#52647a"
PAPER = "#f4f7fb"
BOUNDS = (-4.0, 4.0, -4.0, 4.0)
MAX_LAYERS = 128


def _text(language, ru, en):
    if language not in ("ru", "en"):
        raise ValueError("language must be 'ru' or 'en'")
    return ru if language == "ru" else en


def _finish(fig):
    plt.close(fig)
    return fig


def _rgb_text(color, digits=3):
    return "(" + ", ".join(f"{float(c):.{digits}f}" for c in color) + ")"


def _hex(color):
    return "#" + "".join(f"{round(float(c) * 255):02x}" for c in color)


def _rgb(value):
    return [int(value[k:k + 2], 16) / 255 for k in (1, 3, 5)]


def preset_scene(name="two_spots", language="ru"):
    """Return a fresh front-to-back layer list and an opaque RGB background."""
    red = _text(language, "Красное", "Red")
    blue = _text(language, "Синее", "Blue")
    if name == "two_spots":
        return [
            make_layer(red, scales=(1.5, 1.0), color=(1, 0, 0), opacity=.5),
            make_layer(blue, scales=(1.5, 1.0), color=(0, 0, 1), opacity=.5),
        ], [1.0, 1.0, 1.0]
    if name == "blank":
        return [], [1.0, 1.0, 1.0]
    if name != "flower":
        raise ValueError("Unknown scene preset")
    layers = [make_layer(_text(language, "Середина", "Center"),
                         center=(0, .65), scales=(.43, .43),
                         color=(1, .65, .06), opacity=.98)]
    for i, angle in enumerate((0, 60, 120, 180, 240, 300)):
        radians = np.deg2rad(angle)
        layers.append(make_layer(
            _text(language, f"Лепесток {i + 1}", f"Petal {i + 1}"),
            center=(.95 * np.cos(radians), .65 + .95 * np.sin(radians)),
            scales=(.75, .38), angle_deg=((angle + 180) % 360) - 180,
            color=(.92, .15 + .045 * i, .38 + .035 * i), opacity=.92,
        ))
    layers.extend([
        make_layer(_text(language, "Лист слева", "Left leaf"),
                   center=(-.60, -1.35), scales=(.75, .24), angle_deg=-28,
                   color=(.15, .55, .22), opacity=.92),
        make_layer(_text(language, "Лист справа", "Right leaf"),
                   center=(.58, -1.8), scales=(.7, .24), angle_deg=30,
                   color=(.12, .48, .2), opacity=.92),
        make_layer(_text(language, "Стебель", "Stem"),
                   center=(0, -1.2), scales=(.2, 1.45),
                   color=(.12, .45, .20), opacity=.95),
    ])
    return layers, [1.0, 1.0, 1.0]


def _image(ax, layers, background, title, x=None, y=None, selected=None):
    image = render_layers(layers, background=background, size=161, bounds=BOUNDS)
    # The first/last sample centers are -4 and +4; the outer edges lie half a
    # sample beyond them. This makes the marked grid point match its pixel.
    half_step = (BOUNDS[1] - BOUNDS[0]) / (161 - 1) / 2
    extent = (BOUNDS[0] - half_step, BOUNDS[1] + half_step,
              BOUNDS[2] - half_step, BOUNDS[3] + half_step)
    ax.imshow(image, origin="lower", extent=extent, interpolation="nearest")
    ax.set(xlim=BOUNDS[:2], ylim=BOUNDS[2:], xlabel="x", ylabel="y", title=title)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8, colors=MUTED)
    if x is not None and y is not None:
        ax.plot(x, y, marker="+", color="white", markersize=12, markeredgewidth=3)
        ax.plot(x, y, marker="+", color=INK, markersize=10, markeredgewidth=1.3)
    if selected is not None and 0 <= selected < len(layers):
        layer = layers[selected]
        ax.add_patch(Ellipse(layer["center"], 2 * layer["scales"][0],
                             2 * layer["scales"][1], angle=layer["angle_deg"],
                             fill=False, edgecolor=INK, linewidth=1.2,
                             linestyle="--"))
        ax.plot(*layer["center"], ".", color=INK, markersize=5)
    return image


def overlap_figure(language="ru"):
    """Same two Gaussians and camera; only their compositing order differs."""
    layers, background = preset_scene(language=language)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 5.0))
    fig.subplots_adjust(left=.055, right=.99, bottom=.23, top=.83, wspace=.28)
    titles = _text(language,
                   ["Красное спереди", "Синее спереди", "Разница изображений"],
                   ["Red in front", "Blue in front", "Image difference"])
    a = _image(axes[0], layers, background, titles[0], 0, 0)
    b = _image(axes[1], layers[::-1], background, titles[1], 0, 0)
    difference = np.max(np.abs(a - b), axis=-1)
    axes[2].imshow(difference, origin="lower", extent=BOUNDS,
                   cmap="magma", vmin=0, vmax=.25, interpolation="nearest")
    axes[2].set(title=titles[2], xlabel="x", ylabel="y", xlim=(-4, 4), ylim=(-4, 4))
    axes[2].set_aspect("equal")
    for ax, order in zip(axes[:2], (layers, layers[::-1])):
        result = composite_at(0, 0, order, background)["color"]
        ax.text(.5, -.22, "RGB(0, 0) = " + _rgb_text(result, 2),
                transform=ax.transAxes, ha="center", color=INK, fontsize=10)
    axes[2].text(.5, -.22, _text(language, "Чёрный = одинаково;\nсветлее = больше различие",
                               "Black = identical;\nbrighter = larger difference"),
                 transform=axes[2].transAxes, ha="center", color=INK, fontsize=9)
    fig.suptitle(_text(language, "Поменяли только порядок двух пятен",
                       "Only the order of the two spots changed"), color=INK)
    return _finish(fig)


def alpha_figure(language="ru"):
    """Compare spatial weight with effective alpha without changing the shape."""
    x = np.linspace(-4, 4, 401)
    weight = gaussian2d(x, 0, sx=1.5, sy=1)
    axis = np.linspace(-4, 4, 161)
    xx, yy = np.meshgrid(axis, axis)
    alpha = .5 * gaussian2d(xx, yy, sx=1.5, sy=1)
    fig = plt.figure(figsize=(10.8, 4.6), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1.1])
    ax = fig.add_subplot(grid[0])
    ax.plot(x, weight, color="#2463a9", label="G(x, 0)")
    ax.plot(x, .5 * weight, color="#d85b27", label="α(x, 0) = 0.5 G(x, 0)")
    ax.axvline(1.5, color=MUTED, alpha=.45, linestyle="--")
    ax.set(xlim=(-4, 4), ylim=(0, 1.05), xlabel="x", title=_text(language, "Срез y = 0", "Slice y = 0"))
    ax.grid(alpha=.15)
    ax.legend(fontsize=8, loc="upper right")
    ax2 = fig.add_subplot(grid[1])
    im = ax2.imshow(alpha, origin="lower", extent=BOUNDS, cmap="viridis", vmin=0, vmax=1)
    ax2.set(xlabel="x", ylabel="y", title=_text(language, "Локальная α на плоскости", "Local α on the plane"))
    fig.colorbar(im, ax=ax2, shrink=.72)
    ax3 = fig.add_subplot(grid[2])
    ax3.axis("off")
    rows = _text(language,
                 ["Непрозрачность o = 0.5", "Центр (0, 0):", "G = 1", "α = 0.5 × 1 = 0.5", "",
                  "На одной ширине (1.5, 0):", "G ≈ 0.606531", "α ≈ 0.5 × 0.606531", "   ≈ 0.303265", "",
                  "o — одно число для пятна.", "α — своё число в каждом месте."],
                 ["Opacity o = 0.5", "Center (0, 0):", "G = 1", "α = 0.5 × 1 = 0.5", "",
                  "One scale away (1.5, 0):", "G ≈ 0.606531", "α ≈ 0.5 × 0.606531", "   ≈ 0.303265", "",
                  "o is one number for the spot.", "α varies across the image."])
    ax3.text(0, .97, "\n".join(rows), va="top", color=INK, fontsize=10, linespacing=1.55)
    fig.suptitle(_text(language, "Форма осталась той же, вклад стал слабее",
                       "The shape is unchanged; its contribution is weaker"), color=INK)
    return _finish(fig)


def pixel_figure(layers=None, background=(1, 1, 1), x=0, y=0, language="ru"):
    """Show an image and the numerical front-to-back calculation at one point."""
    if layers is None:
        layers, background = preset_scene(language=language)
    trace = composite_at(x, y, layers, background)
    fig = plt.figure(figsize=(11.2, max(4.5, 2.3 + .38 * len(layers))), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.45])
    ax = fig.add_subplot(grid[0])
    _image(ax, layers, background, _text(language, "Выбранный пиксель", "Selected pixel"), x, y)
    table_ax = fig.add_subplot(grid[1])
    table_ax.axis("off")
    rows = []
    for layer, step in zip(layers, trace["layers"]):
        rows.append([layer["name"], f'{step["weight"]:.3f}', f'{step["alpha"]:.3f}',
                     f'{step["transmittance_before"]:.3f}', _rgb_text(step["contribution"])])
    rows.append([_text(language, "Фон", "Background"), "—", "—",
                 f'{trace["transmittance"]:.3f}',
                 _rgb_text(trace["transmittance"] * np.asarray(background))])
    labels = _text(language, ["Слой", "G", "α", "T до слоя", "Добавили RGB"],
                    ["Layer", "G", "α", "T before", "Added RGB"])
    table = table_ax.table(cellText=rows, colLabels=labels, loc="center",
                           cellLoc="center", colWidths=[.21, .09, .09, .14, .47])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.65)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#dce3ec")
        if r == 0:
            cell.set_facecolor("#e5edf6")
        elif r % 2:
            cell.set_facecolor(PAPER)
    table_ax.text(.5, .95, f"(x, y) = ({x:g}, {y:g})", ha="center", va="top", color=INK)
    table_ax.text(.5, .15, _text(language, "Итоговый RGB: ", "Final RGB: ") + _rgb_text(trace["color"], 4),
                  ha="center", color=INK, fontsize=12)
    table_ax.add_patch(Rectangle((.38, .03), .24, .065, transform=table_ax.transAxes,
                                 facecolor=trace["color"], edgecolor=MUTED))
    fig.suptitle(_text(language, "Сначала передний слой, затем следующий, в конце фон",
                       "Front layer first, then the next layer, then the background"), color=INK)
    return _finish(fig)


def editor_figure(layers=None, background=(1, 1, 1), selected=0,
                  x=0, y=0, outline=True, language="ru"):
    """Editor view with a fixed camera and a selected-layer card."""
    if layers is None:
        layers, background = preset_scene("flower", language)
    trace = composite_at(x, y, layers, background)
    fig = plt.figure(figsize=(10.2, 5.0), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1])
    ax = fig.add_subplot(grid[0])
    _image(ax, layers, background, _text(language, "Изображение", "Image"), x, y,
           selected if outline else None)
    info = fig.add_subplot(grid[1])
    info.axis("off")
    lines = [_text(language, f"Всего пятен: {len(layers)}", f"Spots: {len(layers)}"), ""]
    if selected is not None and 0 <= selected < len(layers):
        layer = layers[selected]
        lines.extend([
            _text(language, f"Выбрано: {layer['name']}", f"Selected: {layer['name']}"),
            _text(language, f"Место в списке: {selected + 1} (первый спереди)", f"List position: {selected + 1} (first is front)"),
            f"(x, y) = {_rgb_text(layer['center'], 2)}",
            f"(s₁, s₂) = {_rgb_text(layer['scales'], 2)}",
            _text(language, f"Поворот: {layer['angle_deg']:.1f}°", f"Rotation: {layer['angle_deg']:.1f}°"),
            f"RGB = {_rgb_text(layer['color'], 3)}",
            _text(language, f"Непрозрачность o = {layer['opacity']:.3f}", f"Opacity o = {layer['opacity']:.3f}"),
        ])
    else:
        lines.append(_text(language, "Пустой холст. Здесь пока только фон.", "Empty canvas. Only the background remains."))
    lines.extend(["", _text(language, f"Проверяем точку ({x:g}, {y:g})", f"Probe at ({x:g}, {y:g})"),
                  "RGB = " + _rgb_text(trace["color"], 4),
                  _text(language, f"До фона осталось T = {trace['transmittance']:.4f}",
                        f"Remaining T at the background = {trace['transmittance']:.4f}")])
    info.text(0, .98, "\n".join(lines), va="top", color=INK, fontsize=10, linespacing=1.6)
    info.add_patch(Rectangle((0, .02), .25, .085, transform=info.transAxes,
                            facecolor=np.clip(trace["color"], 0, 1), edgecolor=MUTED))
    if outline and selected is not None and layers:
        fig.supxlabel(_text(language, "Пунктир: уровень G ≈ 0.607; крестик: проверяемый пиксель.",
                             "Dashed contour: G ≈ 0.607; cross: the probed pixel."), fontsize=9, color=MUTED)
    return _finish(fig)


def _trace_html(layers, background, x, y, language):
    trace = composite_at(x, y, layers, background)
    headings = _text(language, ["Слой: спереди → назад", "G", "o", "α = oG", "T до", "Вклад RGB", "T после"],
                     ["Layer: front → back", "G", "o", "α = oG", "T before", "Added RGB", "T after"])
    rows = []
    for layer, step in zip(layers, trace["layers"]):
        rows.append([html.escape(layer["name"]), f'{step["weight"]:.6g}', f'{layer["opacity"]:.3f}',
                     f'{step["alpha"]:.6g}', f'{step["transmittance_before"]:.6g}',
                     _rgb_text(step["contribution"], 5), f'{step["transmittance_after"]:.6g}'])
    rows.append([_text(language, "Фон", "Background"), "—", "1", "—",
                 f'{trace["transmittance"]:.6g}',
                 _rgb_text(trace["transmittance"] * np.asarray(background), 5), "0"])
    head = "<tr>" + "".join(f"<th>{h}</th>" for h in headings) + "</tr>"
    body = "".join("<tr>" + "".join(f'<td style="padding:5px 8px;border-bottom:1px solid #ddd">{v}</td>'
                                   for v in row) + "</tr>" for row in rows)
    return '<div style="overflow:auto;max-height:320px"><table style="font-size:12px;border-collapse:collapse">' + head + body + "</table></div>"


def gaussian_editor(language="ru"):
    """Build a local editor with layer order, pixel tracing, and JSON round trips.

    ``course_state`` is the mutable scene dictionary. ``course_export()`` returns
    JSON; ``course_import(text)`` validates and replaces a scene atomically.
    """
    import ipywidgets as widgets
    from IPython.display import HTML, clear_output, display

    layers, background = preset_scene(language=language)
    state = {"layers": layers, "background": background}
    busy = [False]
    style = {"description_width": "140px"}
    layout = widgets.Layout(width="445px", max_width="100%")

    def slider(value, lo, hi, step, ru, en):
        return widgets.FloatSlider(value=value, min=lo, max=hi, step=step,
                                   description=_text(language, ru, en), continuous_update=False,
                                   readout_format=".2f", style=style, layout=layout)

    controls = {
        "preset": widgets.Dropdown(options=_text(language,
            [("Красное и синее", "two_spots"), ("Цветок", "flower"), ("Чистый холст", "blank")],
            [("Red and blue", "two_spots"), ("Flower", "flower"), ("Blank canvas", "blank")]),
            value="two_spots", description=_text(language, "Пример", "Example"), style=style, layout=layout),
        "load_preset": widgets.Button(description=_text(language, "Загрузить пример", "Load example"), icon="folder-open"),
        "selected": widgets.Dropdown(description=_text(language, "Выбранное пятно", "Selected spot"), style=style, layout=layout),
        "x": slider(0, -4, 4, .05, "Центр x", "Center x"),
        "y": slider(0, -4, 4, .05, "Центр y", "Center y"),
        "sx": slider(1.5, .2, 3, .05, "Масштаб s₁", "Scale s₁"),
        "sy": slider(1, .2, 3, .05, "Масштаб s₂", "Scale s₂"),
        "angle": slider(0, -180, 180, 1, "Поворот, °", "Rotation, °"),
        "opacity": slider(.5, 0, 1, .05, "Непрозрачность o", "Opacity o"),
        "color": widgets.ColorPicker(value="#ff0000", description=_text(language, "Цвет пятна", "Spot color"), style=style, layout=layout),
        "background": widgets.ColorPicker(value="#ffffff", description=_text(language, "Цвет фона", "Background"), style=style, layout=layout),
        "probe_x": slider(0, -4, 4, .05, "Пиксель x", "Pixel x"),
        "probe_y": slider(0, -4, 4, .05, "Пиксель y", "Pixel y"),
        "outline": widgets.Checkbox(value=True, description=_text(language, "Показать контур выбранного пятна", "Show selected spot's contour"), indent=False),
        "add": widgets.Button(description=_text(language, "Добавить пятно", "Add spot"), icon="plus"),
        "remove": widgets.Button(description=_text(language, "Удалить пятно", "Remove spot"), icon="minus"),
        "front": widgets.Button(description=_text(language, "На слой ближе", "One layer forward"), icon="arrow-up"),
        "back": widgets.Button(description=_text(language, "На слой дальше", "One layer back"), icon="arrow-down"),
        "reset": widgets.Button(description=_text(language, "Сбросить всё", "Reset all"), icon="undo"),
        "export": widgets.Button(description=_text(language, "Выгрузить JSON", "Export JSON"), icon="download"),
        "import": widgets.Button(description=_text(language, "Загрузить JSON", "Import JSON"), icon="upload"),
        "json": widgets.Textarea(value="", placeholder=_text(language, "Здесь появится JSON сцены. Это же поле принимает сохранённый JSON.",
                                                               "Exported scene JSON appears here. Saved scene JSON goes in the same field."),
                                   layout=widgets.Layout(width="100%", height="160px")),
    }
    output = widgets.Output()
    status = widgets.HTML()
    download = widgets.HTML()
    field_names = ("x", "y", "sx", "sy", "angle", "opacity", "color")

    def redraw(change=None):
        if busy[0]:
            return
        with output:
            clear_output(wait=True)
            display(editor_figure(state["layers"], state["background"], controls["selected"].value,
                                  controls["probe_x"].value, controls["probe_y"].value,
                                  controls["outline"].value, language))
            display(HTML(_trace_html(state["layers"], state["background"],
                                     controls["probe_x"].value, controls["probe_y"].value, language)))

    def sync(selected=None):
        busy[0] = True
        options = [(f"{i + 1}. {layer['name']}", i) for i, layer in enumerate(state["layers"])]
        controls["selected"].options = options
        if options:
            index = min(max(0, selected if selected is not None else 0), len(options) - 1)
            controls["selected"].value = index
            layer = state["layers"][index]
            values = dict(x=layer["center"][0], y=layer["center"][1],
                          sx=layer["scales"][0], sy=layer["scales"][1],
                          angle=layer["angle_deg"], opacity=layer["opacity"], color=_hex(layer["color"]))
            for key, value in values.items():
                controls[key].value = value
        else:
            controls["selected"].value = None
        for key in (*field_names, "selected", "remove"):
            controls[key].disabled = not bool(options)
        controls["front"].disabled = not options or controls["selected"].value == 0
        controls["back"].disabled = not options or controls["selected"].value == len(options) - 1
        controls["add"].disabled = len(options) >= MAX_LAYERS
        controls["background"].value = _hex(state["background"])
        busy[0] = False
        redraw()

    def select(change):
        if not busy[0]:
            sync(change["new"])

    def change_layer(change):
        if busy[0] or controls["selected"].value is None:
            return
        index = controls["selected"].value
        old = state["layers"][index]
        # Keep imported RGB values exact until the color picker itself changes.
        color = _rgb(controls["color"].value) if change["owner"] is controls["color"] else old["color"]
        state["layers"][index] = make_layer(
            old["name"], (controls["x"].value, controls["y"].value),
            (controls["sx"].value, controls["sy"].value), controls["angle"].value,
            color, controls["opacity"].value,
        )
        redraw()

    def add(button):
        if len(state["layers"]) >= MAX_LAYERS:
            return
        number = len(state["layers"]) + 1
        names = {layer["name"] for layer in state["layers"]}
        name = _text(language, f"Пятно {number}", f"Spot {number}")
        while name in names:
            number += 1
            name = _text(language, f"Пятно {number}", f"Spot {number}")
        state["layers"].insert(0, make_layer(name, scales=(.7, .7), color=(1, .65, .1), opacity=.7))
        sync(0)

    def remove(button):
        index = controls["selected"].value
        if index is not None:
            state["layers"].pop(index)
            sync(index)

    def move(direction):
        index = controls["selected"].value
        if index is None:
            return
        target = index + direction
        if 0 <= target < len(state["layers"]):
            state["layers"][index], state["layers"][target] = state["layers"][target], state["layers"][index]
            sync(target)

    def set_scene(new_layers, new_background):
        state["layers"] = copy.deepcopy(new_layers)
        state["background"] = list(new_background)
        status.value = ""
        download.value = ""
        sync(0)

    def load_preset(button):
        set_scene(*preset_scene(controls["preset"].value, language))

    def reset(button):
        busy[0] = True
        controls["preset"].value = "two_spots"
        controls["probe_x"].value = controls["probe_y"].value = 0
        controls["outline"].value = True
        controls["json"].value = ""
        busy[0] = False
        set_scene(*preset_scene(language=language))

    def export(button=None):
        value = scene_json(state["layers"], state["background"])
        controls["json"].value = value
        encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
        label = _text(language, "Скачать gaussian-scene.json", "Download gaussian-scene.json")
        download.value = f'<a download="gaussian-scene.json" href="data:application/json;base64,{encoded}">{label}</a>'
        status.value = _text(language, "JSON сцены появился в поле ниже. Текст доступен для копирования, файл — по ссылке.",
                             "The scene is in the text field below and is available as a downloadable file.")
        return value

    def import_scene(value):
        new_layers, new_background = load_scene_json(value)
        if len(new_layers) > MAX_LAYERS:
            raise ValueError(f"The editor supports at most {MAX_LAYERS} spots")
        for layer in new_layers:
            if (not all(-4 <= v <= 4 for v in layer["center"])
                    or not all(.2 <= v <= 3 for v in layer["scales"])
                    or not -180 <= layer["angle_deg"] <= 180):
                raise ValueError("Scene values exceed the editor slider ranges")
        set_scene(new_layers, new_background)

    def import_clicked(button):
        try:
            import_scene(controls["json"].value)
        except (ValueError, TypeError, KeyError):
            status.value = _text(language,
                "<b>Сцена не загружена.</b> Нужен JSON нашего редактора: до 128 пятен, центры от −4 до 4, масштабы от 0.2 до 3, углы от −180° до 180°; RGB и непрозрачность от 0 до 1. Текущая сцена сохранена.",
                "<b>Scene not loaded.</b> This editor accepts its own JSON format: up to 128 spots, centers from −4 to 4, scales from 0.2 to 3, angles from −180° to 180°; RGB and opacity from 0 to 1. The current scene is unchanged.")
        else:
            status.value = _text(language, "Сцена загружена.", "Scene loaded.")

    controls["selected"].observe(select, names="value")
    for key in field_names:
        controls[key].observe(change_layer, names="value")
    for key in ("probe_x", "probe_y", "outline"):
        controls[key].observe(redraw, names="value")

    def change_background(change):
        if not busy[0]:
            state["background"] = _rgb(change["new"])
            redraw()

    controls["background"].observe(change_background, names="value")
    controls["load_preset"].on_click(load_preset)
    controls["add"].on_click(add)
    controls["remove"].on_click(remove)
    controls["front"].on_click(lambda button: move(-1))
    controls["back"].on_click(lambda button: move(1))
    controls["reset"].on_click(reset)
    controls["export"].on_click(export)
    controls["import"].on_click(import_clicked)
    hint = widgets.HTML(_text(language,
        "<b>Первое пятно в списке — переднее.</b> Параметры относятся к выбранному пятну. Крестик отмечает пиксель, чей цвет разобран в таблице. До 128 пятен; картинка обновляется после отпускания ползунка.",
        "<b>The first spot in the list is in front.</b> The controls change the selected spot. The cross marks the pixel calculated in the table. Up to 128 spots; the image updates when a slider is released."))
    layer_controls = widgets.VBox([controls["selected"], controls["x"], controls["y"],
        controls["sx"], controls["sy"], controls["angle"], controls["color"], controls["opacity"]])
    probe_controls = widgets.VBox([controls["background"], controls["probe_x"], controls["probe_y"], controls["outline"]])
    row_layout = widgets.Layout(display="flex", flex_flow="row wrap")
    box = widgets.VBox([
        hint, widgets.HBox([controls["preset"], controls["load_preset"]], layout=row_layout),
        widgets.HBox([layer_controls, probe_controls], layout=row_layout),
        widgets.HBox([controls[k] for k in ("add", "remove", "front", "back", "reset")], layout=row_layout),
        output, widgets.HBox([controls["export"], controls["import"]]), status, download, controls["json"],
    ])
    box.course_controls = controls
    box.course_state = state
    box.course_output = output
    box.course_redraw = redraw
    box.course_export = lambda: scene_json(state["layers"], state["background"])
    box.course_import = import_scene
    sync(0)
    return box
