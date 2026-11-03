"""Notebook experiments for frames, pinhole projection and Gaussian shape."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import ipywidgets as widgets
from IPython.display import display, clear_output

from geometry3d import (
    axis_angle_quaternion, covariance_3d, project_point, projected_covariance,
    projection_jacobian, quaternion_matrix, world_to_camera,
)

BLUE, ORANGE, GREEN = "#365ac0", "#bf623d", "#528165"
PRINCIPAL = (100., 60.)
NEAR = .2
AXES = {"z": (0, 0, 1), "y": (0, 1, 0), "xyz": (1, 1, 1)}


def _rot(axis, angle):
    return quaternion_matrix(axis_angle_quaternion(axis, angle))


def _plane(ax, title, xlabel, ylabel, xlim, ylim):
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, aspect="equal")
    ax.grid(color="#dbe1dc", lw=.65)
    ax.tick_params(labelsize=8)


def frame_figure(language="ru", point_x=1, object_x=1, camera_x=1, yaw=0):
    ru = language == "ru"
    local = np.array([[0, 0, 3], [point_x, 1, 4], [-.6, 0, 4.5], [0, 0, 3]])
    shape = local + [object_x, 0, 0]
    center = np.array([camera_x, 0., 0.])
    basis = _rot([0, 1, 0], yaw)
    camera = np.array([world_to_camera(p, center, basis) for p in shape])
    fig, panels = plt.subplots(1, 2, figsize=(8.2, 4.4), layout="constrained")
    for ax, points, origin, axes, title in zip(
        panels, (shape, camera), (center, np.zeros(3)), (basis, np.eye(3)),
        ("Мир" if ru else "World", "Камера" if ru else "Camera"),
    ):
        _plane(ax, title, "x", "z", (-6, 6), (-2, 7))
        ax.plot(points[:, 0], points[:, 2], color=BLUE)
        ax.scatter(points[1, 0], points[1, 2], color=ORANGE, zorder=4)
        ax.scatter(origin[0], origin[2], color="black", marker="s")
        for j, color in [(0, ORANGE), (2, GREEN)]:
            ax.arrow(origin[0], origin[2], axes[0, j], axes[2, j],
                     color=color, width=.04, length_includes_head=True)
    fig.suptitle("P (x, y, z) = " + str(tuple(np.round(camera[1], 3).tolist())) +
                 (" · в камере" if ru else " · camera frame"))
    return fig


def perspective_figure(language="ru", focal=80, depth=4, camera_z=0):
    ru = language == "ru"
    z = depth - camera_z
    fig, (side, screen) = plt.subplots(1, 2, figsize=(8.4, 4.2), layout="constrained")
    _plane(side, "Лучи сверху" if ru else "Rays from above", "x", "z", (-2.2, 2.2), (-2.5, 8))
    side.plot([.5, 1.5], [depth]*2, lw=4, color=BLUE)
    side.scatter(0, camera_z, c="black", marker="s")
    for x in [.5, 1, 1.5]:
        side.plot([0, x], [camera_z, depth], color=ORANGE, lw=1)
    side.axhline(camera_z+NEAR, color=GREEN, ls=":")
    _plane(screen, "Сенсор" if ru else "Sensor", "u [px]", "v [px]", (-.5, 200.5), (120.5, -.5))
    screen.plot(*PRINCIPAL, "+", c="#555555")
    if z > NEAR:
        a, b = [project_point([x, .5, z], focal, PRINCIPAL, NEAR) for x in [.5, 1.5]]
        screen.plot([a[0], b[0]], [a[1], b[1]], lw=4, color=BLUE)
        screen.scatter(*(a+b)/2, color=ORANGE, zorder=4)
        title = f"z = {z:.2f} m, Δu = {b[0]-a[0]:.2f} px"
    else:
        title = "За ближней плоскостью" if ru else "Clipped at the near plane"
    fig.suptitle(title)
    return fig


def _sphere_lines():
    t = np.linspace(0, 2*np.pi, 97)
    for latitude in np.linspace(-np.pi/2, np.pi/2, 13)[1:-1]:
        yield np.column_stack([np.cos(latitude)*np.cos(t), np.cos(latitude)*np.sin(t),
                               np.full_like(t, np.sin(latitude))])
    for longitude in np.linspace(0, np.pi, 12, endpoint=False):
        yield np.column_stack([np.cos(t)*np.cos(longitude), np.cos(t)*np.sin(longitude), np.sin(t)])


def ellipsoid_figure(language="ru", sx=2, sy=.75, sz=1, angle=45, axis="z"):
    ru = language == "ru"
    rotation = _rot(AXES[axis], angle)
    fig, panels = plt.subplots(1, 2, figsize=(8.4, 4.3), layout="constrained")
    for ax, view, title, labels in zip(panels,
            (_rot([1, 0, 0], -20) @ _rot([0, 1, 0], 30), np.eye(3)),
            (("Косой вид" if ru else "Oblique view"), "XY"),
            (("a", "b"), ("x", "y"))):
        _plane(ax, title, *labels, (-3.1, 3.1), (3.1, -3.1))
        for line in _sphere_lines():
            uvw = (line * [sx, sy, sz]) @ rotation.T @ view.T
            ax.plot(uvw[:, 0], uvw[:, 1], color=BLUE, alpha=.45, lw=.8)
        for i, color in enumerate([ORANGE, GREEN, "#7952a4"]):
            tip = (view @ rotation)[:, i] * [sx, sy, sz][i]
            ax.plot([0, tip[0]], [0, tip[1]], c=color, lw=2)
    fig.suptitle(("Поверхность одного масштаба" if ru else "One-scale surface") +
                 f"   ({sx:g}, {sy:g}, {sz:g})")
    return fig


def footprint_figure(language="ru", depth=4, sz=1, yaw=0, tilt=0):
    ru = language == "ru"
    mu = np.array([1., 0., depth])
    # The camera orbits the world origin; its center keeps the Gaussian at mu in camera space.
    rotation = _rot([0, 1, 0], -yaw) @ _rot([0, 0, 1], tilt)
    scales = np.array([.5, .3, sz])
    cov = rotation @ np.diag(scales**2) @ rotation.T
    uv = project_point(mu, 80, PRINCIPAL, NEAR)
    ellipse = projected_covariance(mu, cov, 80)
    eig, vec = np.linalg.eigh(ellipse)
    fig, ax = plt.subplots(figsize=(7.3, 4.6), layout="constrained")
    _plane(ax, "Кадр 201 × 121" if ru else "201 × 121 frame", "u [px]", "v [px]",
           (-.5, 200.5), (120.5, -.5))
    for line in _sphere_lines():
        p = mu + (line*scales) @ rotation.T
        pix = np.full((len(p), 2), np.nan)
        mask = p[:, 2] > NEAR
        pix[mask] = PRINCIPAL + 80*p[mask,:2]/p[mask,2,None]
        ax.plot(*pix.T, color=ORANGE, alpha=.45, lw=.8)
    theta = np.rad2deg(np.arctan2(vec[1, 1], vec[0, 1]))
    ax.add_patch(Ellipse(uv, 2*np.sqrt(eig[1]), 2*np.sqrt(eig[0]), angle=theta,
                         fill=False, edgecolor=BLUE, lw=2.2,
                         label="Касательная модель" if ru else "Local linear model"))
    ax.plot([], [], c=ORANGE, label="Точная проекция линий" if ru else "Exact projection of mesh lines")
    ax.plot(*uv, "+", color="black")
    ax.legend(loc="upper left", fontsize=8)
    if depth - np.sqrt(cov[2,2]) <= NEAR:
        ax.set_title("Часть поверхности отсечена" if ru else "Part of the surface is clipped")
    return fig


FIGURES = {10: frame_figure, 11: perspective_figure, 12: ellipsoid_figure, 13: footprint_figure}
CONTROLS = {
    10: [("point_x","Точка x","Point x",-2,2,.1,1), ("object_x","Объект x","Object x",-2,2,.1,1),
         ("camera_x","Камера x","Camera x",-2,2,.1,1), ("yaw","Поворот камеры °","Camera yaw °",-70,70,1,0)],
    11: [("focal","Фокус, px","Focal length, px",20,160,1,80), ("depth","Объект z","Object z",1,7,.1,4),
         ("camera_z","Камера z","Camera z",-2,6,.1,0)],
    12: [("sx","Масштаб x","Scale x",.2,3,.05,2), ("sy","Масштаб y","Scale y",.2,3,.05,.75),
         ("sz","Масштаб z","Scale z",.2,3,.05,1), ("angle","Угол °","Angle °",-180,180,1,45)],
    13: [("depth","Глубина центра","Center depth",.5,7,.1,4), ("sz","Масштаб z","Scale z",.1,2,.1,1),
         ("yaw","Ракурс °","View angle °",-90,90,1,0),
         ("tilt","Поворот формы °","Shape rotation °",-90,90,1,0)],
}


def geometry_lab(number, language="ru"):
    """Attach controls to the same figure function used for saved notebook illustrations."""
    controls = {}
    for key, ru, en, lo, hi, step, start in CONTROLS[number]:
        controls[key] = widgets.FloatSlider(value=start, min=lo, max=hi, step=step,
                                            description=ru if language == "ru" else en,
                                            style={"description_width": "initial"}, continuous_update=False)
    if number == 12:
        controls["axis"] = widgets.Dropdown(options=list(AXES), value="z",
                                            description="Ось" if language == "ru" else "Axis")
    output = widgets.Output()

    def draw(_=None):
        fig = FIGURES[number](language, **{key: control.value for key, control in controls.items()})
        with output:
            clear_output(wait=True)
            display(fig)
        plt.close(fig)

    for control in controls.values():
        control.observe(draw, names="value")
    draw()
    return widgets.VBox([*controls.values(), output])
