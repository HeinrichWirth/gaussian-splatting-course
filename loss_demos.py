"""Small image losses and finite-difference experiments for lessons 6 and 7."""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt

BLUE, ORANGE, INK = "#4257d8", "#d66b35", "#26364a"


def _t(language, ru, en):
    if language not in ("ru", "en"):
        raise ValueError("language must be ru or en")
    return ru if language == "ru" else en


def image_errors(prediction, target):
    """Compare corresponding scalar channels, without implicit broadcasting."""
    prediction, target = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("Images must have the same, nonempty shape")
    if not (np.isfinite(prediction).all() and np.isfinite(target).all()):
        raise ValueError("Image values must be finite")
    residual = prediction - target
    return dict(residual=residual, absolute=np.abs(residual), squared=residual ** 2,
                bias=float(residual.mean()), mae=float(np.abs(residual).mean()),
                mse=float(np.mean(residual ** 2)), count=residual.size)


def gaussian_image(mu=-1., sigma=1.2, gain=1., points=33):
    if not np.isfinite([mu, sigma, gain]).all() or sigma <= 0 or not 0 <= gain <= 1:
        raise ValueError("Finite center, positive scale and gain in [0, 1] required")
    if not isinstance(points, int) or points < 2:
        raise ValueError("At least two samples per axis required")
    coordinates = np.linspace(-4, 4, points)
    x, y = np.meshgrid(coordinates, coordinates)
    return gain * np.exp(-.5 * (((x - mu) / sigma) ** 2 + (y / sigma) ** 2))


def bell(mu, sigma=1., gain=1., x=None):
    if not np.isfinite([mu, sigma, gain]).all() or sigma <= 0:
        raise ValueError("Finite parameters and positive scale required")
    x = np.linspace(-4, 4, 81) if x is None else np.asarray(x, dtype=float)
    return gain * np.exp(-.5 * ((x - mu) / sigma) ** 2)


def center_loss(mu, gain=1.):
    return image_errors(bell(mu, gain=gain), bell(1.))["mse"]


def central_difference(function, theta, h=.01):
    if not np.isfinite([theta, h]).all() or h <= 0:
        raise ValueError("Finite parameter and positive difference step required")
    left, right = float(function(theta - h)), float(function(theta + h))
    if not np.isfinite([left, right]).all():
        raise ValueError("Function evaluations must be finite")
    return (right - left) / (2 * h)


def numerical_gradient(function, parameters, h=.001):
    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim != 1 or not len(parameters) or not np.isfinite(parameters).all():
        raise ValueError("Parameters must be a nonempty finite vector")
    result = []
    for index in range(len(parameters)):
        def one_coordinate(value):
            changed = parameters.copy()
            changed[index] = value
            return function(changed)
        result.append(central_difference(one_coordinate, parameters[index], h))
    return np.asarray(result)


def descent_steps(function, initial, rate, steps, h=.01):
    if not np.isfinite([initial, rate]).all() or rate <= 0:
        raise ValueError("Finite initial value and positive learning rate required")
    if not isinstance(steps, int) or not 0 <= steps <= 1000:
        raise ValueError("steps must be an integer between 0 and 1000")
    theta = float(initial)
    history = [(theta, float(function(theta)))]
    for _ in range(steps):
        gradient = central_difference(function, theta, h)
        theta -= rate * gradient
        history.append((theta, float(function(theta))))
    return np.asarray(history)


def _finish(fig):
    plt.close(fig)
    return fig


def _axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
    ax.set_axisbelow(True)


def loss_figure(mu=-1., sigma=1.2, gain=1., metric="absolute", language="ru"):
    if metric not in ("residual", "absolute", "squared"):
        raise ValueError("Unknown map")
    target, prediction = gaussian_image(mu=1), gaussian_image(mu, sigma, gain)
    errors = image_errors(prediction, target)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), layout="constrained")
    titles = [_t(language, "Образец T", "Target T"), _t(language, "Наше изображение P", "Our image P"),
              _t(language, {"residual":"Разность P − T", "absolute":"Модуль |P − T|", "squared":"Квадрат (P − T)²"}[metric],
                 {"residual":"Residual P − T", "absolute":"Absolute |P − T|", "squared":"Squared (P − T)²"}[metric])]
    for ax, values, title in zip(axes, [target, prediction, errors[metric]], titles):
        signed = ax is axes[2] and metric == "residual"
        image = ax.imshow(values, origin="lower", extent=(-4, 4, -4, 4),
                          cmap="RdBu_r" if signed else "magma", vmin=-1 if signed else 0, vmax=1)
        ax.set(title=title, xlabel="x", ylabel="y")
        fig.colorbar(image, ax=ax, orientation="horizontal", fraction=.045, pad=.12)
    fig.suptitle(f"MAE = {errors['mae']:.6f}     MSE = {errors['mse']:.6f}     " +
                 _t(language, "средняя разность", "mean residual") + f" = {errors['bias']:.6f}", fontsize=12)
    return _finish(fig)


def tiny_loss_figure(language="ru"):
    target = np.array([[0., 1.], [.5, .5]])
    prediction = np.array([[.2, .8], [.1, .9]])
    errors = image_errors(prediction, target)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1), layout="constrained")
    titles = _t(language, ["Образец", "Наш результат", "Разность", "Квадрат разности"],
                ["Target", "Prediction", "Residual", "Squared residual"])
    for index, (ax, values, title) in enumerate(zip(axes, [target, prediction, errors["residual"], errors["squared"]], titles)):
        ax.imshow(values, cmap="RdBu_r" if index == 2 else "magma", vmin=-1 if index == 2 else 0, vmax=1)
        for (row, column), value in np.ndenumerate(values):
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", color="black" if abs(value) > .7 else "white",
                    bbox=dict(facecolor="#26364a", alpha=.55, edgecolor="none"))
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    return _finish(fig)


def equal_loss_figure(language="ru"):
    target = np.zeros((8, 8))
    diffuse = np.full((8, 8), .1)
    concentrated = np.zeros((8, 8)); concentrated[3, 3] = .8
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    titles = _t(language, ["Образец: чёрный", "Слабая ошибка везде", "Одна яркая ошибка"],
                ["Target: black", "A faint error everywhere", "One bright error"])
    for ax, value, title in zip(axes, [target, diffuse, concentrated], titles):
        ax.imshow(value, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title + f"\nMSE = {image_errors(value, target)['mse']:.4f}")
        ax.set_xticks([]); ax.set_yticks([])
    return _finish(fig)


def descent_figure(initial=-1., rate=2., steps=0, h=.01, model="gaussian", language="ru"):
    if model not in ("gaussian", "quadratic"):
        raise ValueError("Unknown model")
    function = center_loss if model == "gaussian" else lambda theta: (theta - 3) ** 2
    history = descent_steps(function, initial, rate, steps, h)
    theta, loss = history[-1]
    gradient = central_difference(function, theta, h)
    next_theta = theta - rate * gradient
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    if model == "gaussian":
        x = np.linspace(-4, 4, 81)
        axes[0].plot(x, bell(1), label=_t(language, "Образец: μ = 1", "Target: μ = 1"), color=INK, ls="--")
        axes[0].plot(x, bell(theta), label=f"μ = {theta:.4f}", color=BLUE)
        axes[0].set(xlabel="x", ylabel=_t(language, "Яркость", "Intensity"), xlim=(-4, 4), ylim=(0, 1.05))
    else:
        axes[0].plot(np.arange(len(history)), history[:, 1], "o-", color=BLUE, label="L")
        axes[0].set(xlabel=_t(language, "Шаг", "Step"), ylabel="L")
    axes[0].legend()
    lo, hi = min(-4., theta - 1, next_theta - .5), max(5., theta + 1, next_theta + .5)
    grid = np.linspace(lo, hi, 401)
    axes[1].plot(grid, [function(v) for v in grid], color=BLUE, label="L(θ)")
    tangent_x = np.linspace(theta - .6, theta + .6, 51)
    axes[1].plot(tangent_x, loss + gradient * (tangent_x - theta), color=ORANGE,
                 label=_t(language, "Локальный наклон", "Local slope"))
    axes[1].scatter(history[:, 0], history[:, 1], color=BLUE, s=15, alpha=.6)
    axes[1].scatter([theta], [loss], color=ORANGE, s=55, zorder=4)
    axes[1].scatter([next_theta], [function(next_theta)], facecolors="none", edgecolors=ORANGE, s=65, zorder=4)
    axes[1].set(xlabel="θ = μ" if model == "gaussian" else "θ", ylabel="L", xlim=(lo, hi))
    axes[1].legend()
    for ax in axes: _axes(ax)
    fig.suptitle(_t(language, "Текущий шаг", "Current step") + f" {steps}: θ={theta:.6f}, L={loss:.6f}, g≈{gradient:.6f}\n" +
                 f"θ − ηg = {theta:.6f} − {rate:g} × ({gradient:.6f}) = {next_theta:.6f}", fontsize=11)
    return _finish(fig)


def rates_figure(language="ru"):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    for rate in (.1, .5, .9, 1.1):
        values = descent_steps(lambda theta: (theta-3)**2, 1., rate, 10)
        axes[0].plot(values[:, 0], "o-", ms=3, label=f"η={rate}")
        axes[1].plot(values[:, 1], "o-", ms=3, label=f"η={rate}")
    axes[0].axhline(3, color=INK, ls="--", lw=1)
    axes[0].set(ylabel="θ", xlabel=_t(language,"Шаг", "Step"))
    axes[1].set(ylabel="L = (θ − 3)²", xlabel=_t(language,"Шаг", "Step"))
    for ax in axes: _axes(ax); ax.legend()
    return _finish(fig)


def gradient_figure(mu=-1., gain=.6, language="ru"):
    coordinates = [np.linspace(-3, 3, 81), np.linspace(.1, 1.4, 61)]
    xx, yy = np.meshgrid(*coordinates)
    values = np.array([center_loss(m, a) for m, a in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)
    gradient = numerical_gradient(lambda p: center_loss(p[0], p[1]), [mu, gain])
    fig, ax = plt.subplots(figsize=(7.5, 4.6), layout="constrained")
    image = ax.contourf(xx, yy, values, levels=20, cmap="magma")
    ax.contour(xx, yy, values, levels=10, colors="white", alpha=.3, linewidths=.5)
    ax.scatter([mu, 1], [gain, 1], c=[ORANGE,"white"], s=60)
    ax.arrow(mu, gain, -2*gradient[0], -2*gradient[1], color="cyan", head_width=.06, length_includes_head=True)
    ax.set(xlabel="μ", ylabel="a", title=_t(language,"Ошибка как функция двух параметров", "Loss as a function of two parameters") +
           f"\ngμ≈{gradient[0]:.5f}, ga≈{gradient[1]:.5f}")
    fig.colorbar(image, ax=ax, label="MSE")
    return _finish(fig)


def _lab(controls, figure, language):
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    output = widgets.Output()
    defaults = {key: control.value for key, control in controls.items()}
    def redraw(change=None):
        with output:
            clear_output(wait=True)
            display(figure(**{key: control.value for key, control in controls.items()}, language=language))
    for control in controls.values(): control.observe(redraw, names="value")
    reset = widgets.Button(description=_t(language,"Исходные параметры", "Initial parameters"))
    def reset_values(button):
        for key, value in defaults.items(): controls[key].value = value
    reset.on_click(reset_values)
    box = widgets.VBox([*controls.values(), reset, output])
    box.course_controls, box.course_output, box.course_redraw = controls, output, redraw
    redraw()
    return box


def loss_lab(language="ru"):
    import ipywidgets as widgets
    style = {"description_width":"initial"}
    def slider(value, minimum, maximum, step, title):
        return widgets.FloatSlider(value=value,min=minimum,max=maximum,step=step,description=title,
                                   continuous_update=False,style=style,layout=widgets.Layout(width="95%"))
    return _lab(dict(mu=slider(-1,-3,3,.1,_t(language,"Центр μ", "Center μ")),
                     sigma=slider(1.2,.3,2.5,.1,_t(language,"Масштаб σ", "Scale σ")),
                     gain=slider(1,0,1,.05,_t(language,"Яркость a", "Intensity a")),
                     metric=widgets.Dropdown(options=_t(language,[("Модуль","absolute"),("Квадрат","squared"),("Со знаком","residual")],
                         [("Absolute","absolute"),("Squared","squared"),("Signed","residual")]), description=_t(language,"Карта ошибки","Error map"),style=style)),loss_figure,language)


def descent_lab(language="ru"):
    import ipywidgets as widgets
    style = {"description_width":"initial"}
    def slider(value, minimum, maximum, step, title):
        return widgets.FloatSlider(value=value,min=minimum,max=maximum,step=step,description=title,
                                   continuous_update=False,style=style,layout=widgets.Layout(width="95%"))
    return _lab(dict(initial=slider(-1,-4,4,.1,_t(language,"Начальный центр μ", "Initial center μ")),
                     rate=slider(2,.05,12,.05,_t(language,"Размер шага η", "Learning rate η")),
                     steps=widgets.IntSlider(value=0,min=0,max=40,description=_t(language,"Число шагов","Number of steps"),style=style,continuous_update=False),
                     h=widgets.Dropdown(options=[.001,.01,.1,.5],value=.01,description="h",style=style)),descent_figure,language)
