"""Moving cameras, a changing Gaussian, and a pair of persistent neighbours."""
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output

import dynamic_motion as motion
from splat_render import make_scene, render_scene, translated_camera


def motion_figure(number, language='ru', **values):
    def t(ru, en):
        return en if language == 'en' else ru
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.8, 3.6), layout='constrained')
    if number == 22:
        views, _, estimate = motion.observing_pair(**values)
        for ax, view, name in zip([left, right], views, ['A', 'B']):
            ax.imshow(motion.observation_image(view, values.get('gate', False)))
            ax.set_title(f"{name}   t = {view['time']:.2f} s")
            ax.set_xlabel('u (px)')
            ax.set_ylabel('v (px)')
        status = t('нет пары', 'no pair') if estimate is None else f'{estimate:.2f} m'
        fig.suptitle(t('Глубина из двух отметок', 'Depth from two marks')+'   '+status)
    elif number == 23:
        time = values.get('time', 1)
        middle, noise, bend = [values.get(k, d) for k, d in [('middle', False), ('noise', 0.), ('bend', .2)]]
        ts = np.linspace(0, 3, 121)
        left.axvspan(2, 3, color='#efe7d6')
        left.plot(ts, motion.true_x(ts, bend), color='#467c62', label=t('Истинное движение', 'Reference motion'))
        left.plot(ts, motion.fit_path(ts, middle, noise, bend), '--', color='#324ba5', label=t('Модель', 'Fit'))
        times = [0, 1, 2] if middle else [0, 2]
        left.scatter(times, [motion.true_x(k, bend) + (noise if k == 1 else 0) for k in times], color='#263345')
        left.axvline(time, color='#ad6733', linewidth=1)
        left.legend(fontsize=8); left.grid(alpha=.2);left.set(xlabel='t (s)', ylabel='x (m)', xlim=(0, 3))
        center = motion.fit_path(time, middle, noise, bend)
        image = render_scene(make_scene(motion.trajectory_splat(time, center)), translated_camera(0, motion.F),
                             motion.WIDTH, motion.HEIGHT)
        right.imshow(image)
        right.axvline(motion.horizontal_pixel(motion.true_x(time, bend), 0), color='#467c62', linestyle=':')
        right.set(xlabel='u (px)', ylabel='v (px)', title=t('Пятно модели и центр образца', 'Fitted splat and reference center'))
    elif number == 24:
        pair = motion.persistent_pair(**values)
        for ax, key, title in [(left,'observed',t('Одна отметка','Measurement alone')),
                               (right,'fitted',t('Отметка и соседство','Measurement and neighbour'))]:
            for k, c, name in [('rigid','#4a78bb',t('Поворот A','Rotation of A')),
                               ('actual','#548267',t('Истинный B','True B')),
                               (key,'#cf6835',t('Оценка B','Estimated B'))]:
                x, y = pair[k]; ax.plot([0, x], [0, y], color=c, linestyle=':' if k=='rigid' else '-', alpha=.8)
                ax.scatter([x], [y], color=c, marker='x' if k=='actual' else 'o', label=name)
            ax.scatter([0], [0], color='#333'); ax.text(.035, -.05, 'A')
            ax.legend(fontsize=7, loc='lower left')
            ax.set(xlim=(-1.4,1.5), ylim=(1.4,-1.4), xlabel='x (m)', ylabel='y (m)', title=title)
            ax.set_aspect('equal'); ax.grid(alpha=.2)
    else:
        raise ValueError('motion figures cover lessons 22–24')
    return fig


def motion_lab(number, language='ru'):
    def t(ru, en):
        return en if language == 'en' else ru
    def slider(ru, en, value, low, high, step):
        return widgets.FloatSlider(value=value,min=low,max=high,step=step,description=t(ru,en),
                                   style={'description_width':'initial'},continuous_update=False)
    def toggle(ru, en):
        return widgets.Checkbox(value=False,description=t(ru,en),indent=False)
    if number == 22:
        controls = dict(time=slider('Время A, с','A time, s',.8,0,2,.05),
                        object_speed=slider('Скорость пятна, м/с','Splat speed, m/s',1,-1,1,.1),
                        camera_speed=slider('Скорость камер, м/с','Rig speed, m/s',0,-1,1,.1),
                        lag=slider('Задержка B, с','B delay, s',0,0,.8,.05),gate=toggle('Ширма','Occluder'))
    elif number == 23:
        controls = dict(time=slider('Время, с','Time, s',1,0,3,.05),
                        bend=slider('Коэффициент изгиба, м/с²','Quadratic term, m/s²',.2,-.25,.25,.01),
                        middle=toggle('Третий замер','Third observation'),
                        noise=slider('Сдвиг замера, м','Middle error, m',0,-.2,.2,.01))
    elif number == 24:
        controls = dict(angle=slider('Поворот A, °','Rotation of A, °',90,0,150,5),
                        strength=slider('Вес соседства','Neighbour weight',4,0,10,.25),
                        noise=slider('Ошибка отметки по x, м','Mark error in x, m',.4,-.5,.5,.025),
                        bend=slider('Сгиб, °','Bend, °',0,-60,60,5))
    else:
        raise ValueError('motion labs cover lessons 22–24')
    output = widgets.Output()
    def update(change=None):
        with output:
            clear_output(wait=True)
            fig = motion_figure(number, language, **{k:w.value for k,w in controls.items()})
            display(fig)
            plt.close(fig)
    for widget in controls.values():
        widget.observe(update, names='value')
    update()
    return widgets.VBox([*controls.values(), output])
