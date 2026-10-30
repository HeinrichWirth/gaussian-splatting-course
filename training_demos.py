"""Figures and local notebook controls for the learning experiments."""
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np

from training_math import chain_example, fit_one_gaussian, fit_image, make_target


def _text(language, ru, en):
    if language not in ("ru", "en"):
        raise ValueError("language must be ru or en")
    return ru if language == "ru" else en


def chain_figure(mu=0, language="ru"):
    values = chain_example(mu=mu)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axis = np.linspace(-2, 2.5, 300)
    losses = [chain_example(mu=v)["loss"] for v in axis]
    axes[0].plot(axis, losses, color="#4358cf")
    axes[0].scatter([mu], [values["loss"]], color="#ed8c51", zorder=3)
    axes[0].plot([mu - .35, mu + .35],
                 [values["loss"] - .35 * values["gradient"],
                  values["loss"] + .35 * values["gradient"]], color="#ed8c51")
    axes[0].set(xlabel="μ", ylabel="L", title=_text(language, "Ошибка одного измерения", "One-sample loss"))
    axes[0].grid(alpha=.15)
    axes[1].axis("off")
    labels = ["d = 1 − μ", "u = d / 1", "q = u × u", "t = −q / 2", "G = exp(t)", "r = G − 0.8", "L = r × r"]
    nums = [values[k] for k in ["d", "u", "q", "t", "weight", "residual", "loss"]]
    for i, (label, value) in enumerate(zip(labels, nums)):
        axes[1].text(.04, .96 - i * .12, f"{label:15} = {value:.6f}", family="monospace", fontsize=11)
    axes[1].text(.04, .04, f"dL/dμ = {values['gradient']:.6f}", color="#4358cf", fontsize=12)
    plt.close(fig)
    return fig


def learning_figure(result=None, language="ru"):
    result = fit_one_gaussian() if result is None else result
    history = result["history"]
    x = result["x"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    axes[0].plot(x, result["target"], color="#263d43", label=_text(language, "Образец", "Target"))
    axes[0].plot(x, np.exp(-.5 * (x - history[0]["center"]) ** 2), "--", color="#aaa", label=_text(language, "Начало", "Start"))
    axes[0].plot(x, result["prediction"], color="#4358cf", label=_text(language, "После обновлений", "After updates"))
    axes[0].set(xlabel="x", ylabel="G(x)")
    axes[0].legend(fontsize=8)
    axes[1].plot([h["step"] for h in history], [h["loss"] for h in history], color="#4358cf")
    axes[1].set(xlabel=_text(language, "Шаг", "Step"), ylabel="MSE")
    axes[2].plot([h["step"] for h in history], [h["center"] for h in history], color="#ed8c51")
    axes[2].axhline(1.5, linestyle="--", color="#263d43")
    axes[2].set(xlabel=_text(language, "Шаг", "Step"), ylabel="μ")
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle(f"μ: {history[0]['center']:.3f} → {history[-1]['center']:.6f}; MSE: {history[0]['loss']:.6f} → {history[-1]['loss']:.8f}")
    plt.close(fig)
    return fig


def autograd_lab(language="ru"):
    import ipywidgets as widgets
    from IPython.display import clear_output, display
    initial = widgets.FloatSlider(value=-1, min=-3, max=3, step=.1,
                                  description=_text(language, "Начальная μ", "Initial μ"), continuous_update=False)
    rate = widgets.FloatSlider(value=2, min=0, max=20, step=.1, description="η", continuous_update=False)
    steps = widgets.IntSlider(value=0, min=0, max=100, step=1,
                              description=_text(language, "Шаг", "Step"), continuous_update=False)
    one = widgets.Button(description=_text(language, "+1 шаг", "+1 step"))
    reset = widgets.Button(description=_text(language, "Начало", "Reset"))
    output = widgets.Output()

    def refresh(_=None):
        result = fit_one_gaussian(initial.value, rate=rate.value, steps=steps.value)
        with output:
            clear_output(wait=True)
            display(learning_figure(result, language))

    for field in (initial, rate, steps):
        field.observe(refresh, names="value")
    one.on_click(lambda _: setattr(steps, "value", min(steps.max, steps.value + 1)))
    reset.on_click(lambda _: setattr(steps, "value", 0))
    refresh()
    return widgets.VBox([widgets.HBox([initial, rate]), widgets.HBox([steps, one, reset]), output])


@lru_cache(maxsize=8)
def demo_fit(target="shapes", count=64, steps=120, placement="grid", scale=.55, seed=7, rate_multiplier=1):
    return fit_image(make_target(target), count=count, steps=steps, placement=placement,
                     scale=scale, seed=seed, rate_multiplier=rate_multiplier)


def fitting_figure(result=None, language="ru"):
    result = demo_fit() if result is None else result
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), layout="constrained")
    images = [result["target"], result["snapshots"][0], result["image"]]
    labels = _text(language, ["Образец", "Начало", "После обучения"], ["Target", "Start", "After training"])
    for ax, image, label in zip(axes, images, labels):
        ax.imshow(image, origin="lower", extent=(-4, 4, -4, 4), interpolation="nearest")
        ax.set(title=label, xticks=[-4, 0, 4], yticks=[-4, 0, 4])
    axes[3].plot(result["history"], color="#4358cf")
    axes[3].set(xlabel=_text(language, "Шаг", "Step"), ylabel="MSE")
    axes[3].grid(alpha=.15)
    fig.suptitle(f"N = {result['settings']['count']}; MSE: {result['history'][0]:.6f} → {result['history'][-1]:.6f}")
    plt.close(fig)
    return fig


def detail_figure(result, language="ru"):
    difference = np.mean(np.abs(result["image"] - result["target"]), axis=-1)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), layout="constrained")
    im = axes[0].imshow(difference, origin="lower", extent=(-4, 4, -4, 4), cmap="magma", vmin=0, vmax=1)
    axes[0].set(title=_text(language, "Средняя абсолютная ошибка RGB", "Mean absolute RGB error"))
    fig.colorbar(im, ax=axes[0], shrink=.7)
    axes[1].imshow(result["image"], origin="lower", extent=(-4, 4, -4, 4), interpolation="nearest")
    centers = result["parameters"]["centers"].detach().numpy()
    axes[1].scatter(centers[:, 0], centers[:, 1], s=8, color="black")
    axes[1].set(title=_text(language, "Где оказались центры", "Where the centers ended up"), xlim=(-4, 4), ylim=(-4, 4))
    plt.close(fig)
    return fig


def fitting_lab(language="ru"):
    import ipywidgets as widgets
    from IPython.display import clear_output, display
    target = widgets.Dropdown(options=_text(language, [("Фигуры", "shapes"), ("Цветок", "flower"), ("Клетки", "checker")], [("Shapes", "shapes"), ("Flower", "flower"), ("Checker", "checker")]), description=_text(language, "Образец", "Target"))
    count = widgets.Dropdown(options=[16, 32, 64, 128], value=64, description="N")
    placement = widgets.Dropdown(options=_text(language, [("Сетка", "grid"), ("Случайно", "random"), ("Около центра", "center")], [("Grid", "grid"), ("Random", "random"), ("Near center", "center")]), description=_text(language, "Начало", "Placement"))
    scale = widgets.FloatSlider(value=.55, min=.2, max=3, step=.05, description="σ₀", continuous_update=False)
    steps = widgets.IntSlider(value=120, min=0, max=400, step=20, description=_text(language, "Шагов", "Steps"), continuous_update=False)
    rate = widgets.FloatSlider(value=1, min=0, max=4, step=.1, description=_text(language, "Множитель η", "Rate factor"), continuous_update=False)
    run = widgets.Button(description=_text(language, "Обучение", "Train"), button_style="primary")
    export = widgets.Textarea(description="JSON", layout=widgets.Layout(width="100%", height="100px"))
    output = widgets.Output()

    def refresh(_=None):
        run.disabled = True
        try:
            result = demo_fit(target.value, count.value, steps.value, placement.value, scale.value, 7, rate.value)
            with output:
                clear_output(wait=True)
                display(fitting_figure(result, language))
            export.value = result["scene"]
        finally:
            run.disabled = False

    run.on_click(refresh)
    refresh()
    return widgets.VBox([widgets.HBox([target, count, placement]), widgets.HBox([scale, steps, rate]), run, output,
                         widgets.HTML(_text(language, "Параметры сцены в формате редактора из урока 5.", "Scene parameters in the lesson 5 editor format.")), export])
