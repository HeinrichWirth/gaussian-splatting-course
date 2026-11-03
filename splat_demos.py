"""Figures and notebook controls for the renderer and density experiments."""
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display

import splat_render as sr


def _labels(lang, ru, en):
    return ru if lang == 'ru' else en


def panel(images, titles, note=None):
    fig, axes = plt.subplots(1, len(images), figsize=(3.1*len(images), 3.05),
                             layout="constrained", squeeze=False)
    for ax, image, title in zip(axes[0], images, titles):
        ax.imshow(np.clip(image, 0, 1), interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    if note:
        fig.suptitle(note, fontsize=10)
    return fig


def visibility_figure(lang, angle=0., depth=3.5, u=32, visible="both", background="white"):
    scene, cam = sr.overlap_scene(depth), sr.orbit_camera(angle)
    if visible != "both":
        scene["splats"] = [scene["splats"][int(visible)]]
    bg = (1.,)*3 if background == "white" else (0.,)*3
    a = sr.render_scene(scene, cam, background=bg)
    b = sr.render_scene(scene, cam, background=bg, reverse=True)
    fig = panel([a,b],_labels(lang,['По глубине','Обратный порядок'],['Depth order','Reversed order']))
    for ax in fig.axes:
        ax.plot(u,32,marker='+',color='black',markersize=8)
    remaining = 1.
    parts = []
    for spot in sr.project_splats(scene, cam, 65, 65):
        alpha = sr.footprint_alpha((u, 32), spot)
        label = _labels(lang, 'красное', 'red') if spot["color"][0] else _labels(lang, 'синее', 'blue')
        parts.append(f'{label} {remaining*alpha:.3f}')
        remaining *= 1-alpha
    parts.append(f'{_labels(lang, "фон", "background")} {remaining:.3f}')
    fig.suptitle(f'({u}, 32) · RGB = '+np.array2string(a[32,u],precision=3)+'\n'+'  ·  '.join(parts), fontsize=10)
    return fig


def visibility_profile(lang):
    fig, ax = plt.subplots(figsize=(6.8,3),layout='constrained')
    for item, color, label in zip(sr.project_splats(sr.overlap_scene(),sr.orbit_camera(),65,65),
                                  ('#ce4847','#4468c2'),_labels(lang,['Красное','Синее'],['Red','Blue'])):
        u = np.arange(65)
        ax.plot(u, sr.footprint_alpha(np.column_stack((u,np.full_like(u,32))),item),
                color=color,label=label)
    ax.set(xlabel='u (px)',ylabel='α',ylim=(0,1));ax.legend();return fig


def harmonic_figure(lang, angle=-36.87, elevation=0., cx=.6, cy=0., cz=.4):
    a,b=np.deg2rad([angle,elevation])
    direction=np.array([np.sin(a)*np.cos(b),np.sin(b),np.cos(a)*np.cos(b)])
    right=[np.cos(a),0,-np.sin(a)]
    down=[-np.sin(a)*np.sin(b),np.cos(b),-np.cos(a)*np.sin(b)]
    cam=dict(center=np.array([0,0,4])-4*direction,rotation=np.column_stack((right,down,direction)),focal=65)
    dot=sr.splat([0,0,4],[.65]*3,[.5]*3,1)
    coefficients=[[0,0,0],[cy,0,0],[cz,0,0],[cx,0,0]]
    images=[sr.render_scene(sr.make_scene(dot),cam),
            sr.render_scene(sr.make_scene(dict(dot,sh=coefficients)),cam)]
    return panel(images,_labels(lang,['Постоянный цвет','Цвет с направлением'],['Constant color','Directional color']),
                 f'RGB = {np.round(sr.sh_color(direction,coefficients),3)}')


def harmonic_curve(lang):
    angles=np.linspace(-180,180,241)
    rad=np.deg2rad(angles)
    cx=-.6*sr.C1*np.sin(rad);cz=.4*sr.C1*np.cos(rad)
    fig,ax=plt.subplots(figsize=(7,3.3),layout='constrained')
    ax.plot(angles,cx,label='−0.6 C₁ x');ax.plot(angles,cz,label='+0.4 C₁ z')
    ax.plot(angles,.5+cx+cz,label='R',color='#bb5144',lw=2.5)
    ax.axhline(0,color='#888',lw=.6)
    ax.set(xlabel=_labels(lang,'Азимут (°)','Azimuth (°)'),ylabel=_labels(lang,'Вклад / цвет','Contribution / color'))
    ax.legend(ncol=3,fontsize=8);return fig


def ray_figure(lang, depth=2.4, views=2, steps=0, baseline=.8):
    train=(0,.8) if views==2 else (0,)
    for _ in range(steps):
        depth,_=sr.depth_step(depth,train)
    cam=sr.translated_camera(baseline)
    images=[sr.render_scene(sr.ray_scene(z),cam,33,33) for z in (4,depth)]
    return panel(images,_labels(lang,['Образец','Модель'],['Target','Model']),
                 f'Z = {depth:.4f} · MSE = {sr.ray_loss(depth,train):.7f}')


def ray_landscape(lang):
    depths=np.linspace(1.4,7,72)
    fig,ax=plt.subplots(figsize=(7,3.15),layout='constrained')
    for x,label in zip((0,.8,-.6),_labels(lang,['Камера A','Камера B','Отложенная'],['Camera A','Camera B','Held out'])):
        ax.plot(depths,[sr.ray_loss(z,(x,)) for z in depths],label=label)
    ax.axvline(4,color='#888',ls=':');ax.set(xlabel='Z',ylabel='MSE');ax.legend();return fig


def density_figure(lang, operation="initial", selected=0, steps=0, threshold=.001, pixel=16):
    scene=sr.density_scene()
    if operation=="adaptive":
        scene=sr.density_pass(scene,threshold)
    elif operation!="initial":
        scene=sr.density_edit(scene,operation,selected)
    for _ in range(steps):
        scene=sr.density_step(scene)
    cam=sr.translated_camera(0,42)
    images=[sr.render_scene(s,cam,33,33) for s in (sr.density_scene(True),scene)]
    target_red, model_red = (image[16, pixel, 0] for image in images)
    titles = _labels(lang, ['Образец', 'Модель'], ['Target', 'Model'])
    fig = panel(images, [f"{title} · R = {red:.3f}" for title, red in
                         zip(titles, (target_red, model_red))],
                f'N = {len(scene["splats"])} · MSE = {sr.density_loss(scene):.7f}\n'
                f'(u, v) = ({pixel}, 16) · ΔR = {model_red-target_red:+.3f}')
    for ax in fig.axes:
        ax.plot(pixel, 16, marker="s", markersize=10, markerfacecolor="none",
                markeredgecolor="#172a40", markeredgewidth=1.2)
    return fig


def density_chart(lang):
    scene=sr.density_scene()
    candidates=[scene,sr.density_step(scene),*(sr.density_edit(scene,op) for op in ('clone','split','prune','reset'))]
    labels=_labels(lang,['Старт','Шаг','Клон','Деление','Удаление','Сброс'],
                        ['Start','Step','Clone','Split','Prune','Reset'])
    fig,ax=plt.subplots(figsize=(7.4,3),layout='constrained')
    ax.bar(labels,[sr.density_loss(s) for s in candidates],color=['#778282','#5874bc','#b37b56','#71987c','#9b8cac','#cb7569'])
    ax.set_ylabel('MSE');return fig


def experiment(number,lang):
    def tr(ru,en): return _labels(lang,ru,en)
    def slider(name,value,lo,hi,step):
        kind=widgets.IntSlider if all(isinstance(s,int) for s in (value,lo,hi,step)) else widgets.FloatSlider
        return kind(description=name,value=value,min=lo,max=hi,step=step,continuous_update=False,
                    style={'description_width':'initial'})
    def menu(name,values):
        return widgets.Dropdown(description=name,options=values,style={'description_width':'initial'})
    if number==14:
        draw=visibility_figure
        controls=dict(angle=slider(tr('Угол','Angle'),0,-140,140,1),depth=slider(tr('Красное Z','Red Z'),3.5,2,6,.05),
                      u=slider('u',32,0,64,1),visible=menu(tr('Пятна','Splats'),[(tr('Оба','Both'),'both'),(tr('Красное','Red'),'0'),(tr('Синее','Blue'),'1')]),
                      background=menu(tr('Фон','Background'),[(tr('Белый','White'),'white'),(tr('Чёрный','Black'),'black')]))
    elif number==15:
        draw=harmonic_figure
        controls=dict(angle=slider(tr('Азимут','Azimuth'),-36.87,-180,180,.1),elevation=slider(tr('Наклон','Elevation'),0,-80,80,1),
                      cx=slider('−x',.6,-2,2,.05),cy=slider('−y',0,-2,2,.05),cz=slider('+z',.4,-2,2,.05))
    elif number==16:
        draw=ray_figure
        controls=dict(views=menu(tr('Камер','Cameras'),[2,1]),depth=slider('Z',2.4,1.25,7,.05),
                      steps=slider(tr('Шагов','Steps'),0,0,30,1),baseline=menu(tr('Ракурс','Preview'),[.8,0,-.6]))
    else:
        draw=density_figure
        options=list(zip(tr(['Начало','Клон','Деление','Удаление','Сброс','Автопроверка'],
                            ['Initial','Clone','Split','Prune','Reset','Adaptive']),
                         ['initial','clone','split','prune','reset','adaptive']))
        controls=dict(operation=menu(tr('Действие','Action'),options),selected=menu(tr('Пятно','Splat'),[0,1]),
                      steps=slider(tr('Шагов после','Steps after edit'),0,0,12,1),threshold=slider(tr('Порог','Threshold'),.001,0,.025,.0001),
                      pixel=slider(tr('Столбец u (строка 16)','Column u (row 16)'),16,0,32,1))
    def redraw(**settings):
        fig=draw(lang,**settings);display(fig);plt.close(fig)
    output=widgets.interactive_output(redraw,controls)
    return widgets.VBox([*controls.values(),output])
