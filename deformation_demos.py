"""Notebook views of the sampled deformation field and the fitted ribbon."""
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output

from deformation import (TRAIN_TIMES, field_scene, field_value, reference_scene, ribbon,
                         example_model, scene_at, stereo_tracks)
from splat_render import orbit_camera, render_scene


def frame(scene, angle=0):
    return render_scene(scene, orbit_camera(angle, 64), 65, 49)


def field_figure(time=.5, angle=0., corner=1., index=4, lang='ru'):
    ru = lang == 'ru'
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), layout='constrained')
    for ax, scene, title in zip(axes[:2], [ribbon(),field_scene(time, corner)],
                                ['Исходная лента','В этот момент'] if ru else ['Rest ribbon','At this time']):
        ax.imshow(frame(scene, angle)); ax.set_title(title); ax.axis('off')
    u = (ribbon()['splats'][index]['center'][0]+1)/2
    grid = np.linspace(0,1,101)
    values = corner*grid[:,None]*grid[None,:]
    axes[2].imshow(values, origin='lower', extent=[0,1,0,1], cmap='BrBG', vmin=-1, vmax=1)
    axes[2].plot([0,u,u],[time,time,0], color='#8b3540', lw=1)
    axes[2].plot(u,time,'o',color='#8b3540')
    axes[2].set(xlabel='u',ylabel='t (s)',xticks=[0,.5,1],yticks=[0,.5,1],
                title=f'F = {field_value(u, time, corner):.3f}')
    return fig


def fit_figure(time=.375, angle=32., noise=0., static=False, lang='ru'):
    ru = lang == 'ru'
    model = example_model(noise)
    if static:
        model['amplitudes'] = [[0,0,0] for _ in model['amplitudes']]
    target = frame(reference_scene(time),angle)
    output = frame(scene_at(model,time),angle)
    fig,axes = plt.subplots(1,3,figsize=(9,3),layout='constrained')
    labels = ['Эталон','Модель','Разность RGB'] if ru else ['Reference','Model','RGB difference']
    delta = np.abs(output-target).mean(axis=2)
    axes[0].imshow(target); axes[1].imshow(output)
    im = axes[2].imshow(delta,vmin=0,vmax=.2,cmap='magma')
    fig.colorbar(im,ax=axes[2],shrink=.66)
    for ax,title in zip(axes,labels):
        ax.axis('off'); ax.set_title(title)
    fig.suptitle(f't = {time:.3f} s     θ = {angle:g}°     MSE = {np.mean((target-output)**2):.6f}')
    return fig


def motion_plot(noise=0., lang='ru'):
    ru = lang == 'ru'
    model = example_model(noise)
    times = np.linspace(0,1,101)
    actual = [reference_scene(t)['splats'][-1]['center'][1] for t in times]
    fitted = [scene_at(model,t)['splats'][-1]['center'][1] for t in times]
    raw = stereo_tracks(noise)
    recovered = []
    for row in raw:
        (ul,v),(ur,_) = row['left'][-1], row['right'][-1]
        recovered.append((v-24)*.8/(ul-ur))
    fig,ax = plt.subplots(figsize=(6.3,3.1),layout='constrained')
    ax.plot(times,actual,color='#287d75',label='Съёмка' if ru else 'Reference')
    ax.plot(times,fitted,'--',color='#bb672d',label='Модель' if ru else 'Fitted')
    ax.scatter(TRAIN_TIMES,recovered,color='#343a43',zorder=3,label='Две метки' if ru else 'Two observations')
    ax.axvline(.375,ls=':',color='#576c97',label='Отложенное время' if ru else 'Held-out time')
    ax.set(xlabel='t (s)',ylabel='y (m)'); ax.grid(alpha=.2); ax.legend(ncol=2,fontsize=8)
    return fig


def lab(number, lang='ru'):
    ru = lang == 'ru'
    def slider(name, value, low, high, step):
        return widgets.FloatSlider(description=name,value=value,min=low,max=high,step=step,
                                   style={'description_width':'initial'},continuous_update=True)
    controls = {'time':slider('Время, с' if ru else 'Time, s',.5 if number==25 else .375,0,1,.005),
                'angle':slider('Камера, °' if ru else 'Camera, °',0 if number==25 else 32,-65,65,1)}
    if number == 25:
        controls['corner'] = slider('Верхний узел' if ru else 'Upper node',1,-1,1,.05)
        controls['index'] = widgets.Dropdown(description='Пятно' if ru else 'Splat',
                                             options=[(str(i+1),i) for i in range(5)],value=4)
    else:
        controls['noise'] = slider('Ошибка метки, px' if ru else 'Landmark error, px',0,0,1,.05)
        controls['static'] = widgets.Checkbox(description='Неподвижная лента' if ru else 'Static ribbon')
    out = widgets.Output()
    figure = field_figure if number == 25 else fit_figure
    def update(*_):
        with out:
            clear_output(wait=True)
            fig = figure(**{key:item.value for key,item in controls.items()},lang=lang)
            display(fig)
            plt.close(fig)
    for widget in controls.values():
        widget.observe(update,names='value')
    update()
    return widgets.VBox([*controls.values(),out])
