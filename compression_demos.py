"""Figures and live Jupyter controls for lessons 27–30."""
from functools import lru_cache

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from compression_math import (DEFAULT_STUDY, benchmark_photo, choose_candidate, memory_budget, motion_data,
                             pack_motion, pack_static, photo_data, pose_with_centers, sample_track,
                             static_candidates, unpack_motion, unpack_static)
from deformation import scene_at
from reconstruction import benchmark, scores
from splat_render import orbit_camera, render_scene


@lru_cache(maxsize=1)
def _photos():
    return photo_data()


@lru_cache(maxsize=1)
def _motion():
    return motion_data()


def storage_figure(lang='en',count=117,degree=0,width=80):
    t=lambda ru,en:en if lang=='en' else ru
    height=max(1,int(np.floor(width*.75+.5)))
    b=memory_budget(count,degree,width,height)
    fig,ax=plt.subplots(figsize=(7.4,2.9),layout='constrained')
    labels=[t('Параметры','Parameters'),t('Один RGB-кадр','One RGB frame'),t('Параметры + кадр','Weights + frame'),
            t('Параметры + градиент + Adam','Weights + gradient + Adam')]
    v=[b['weights'],b['rgb'],b['viewer'],b['adam']]
    bars=ax.barh(labels,v,color=['#467184','#b26d43','#728644','#876d95'])
    ax.bar_label(bars,fmt='%d',padding=3);ax.set_xlim(0,max(v)*1.20)
    ax.invert_yaxis();ax.set_xlabel(t('байты','bytes'))
    ax.set_title(f'{count} · L = {degree} · {width} × {height}');return fig


def static_figure(lang='en',keep=100,precision='u8',color='rgb',view='view-04.jpg'):
    scene,frames=_photos();f=frames[view]
    packet=pack_static(scene,precision,keep,color)
    model=unpack_static(packet);w,h=80,60
    source=render_scene(scene,f['camera'],w,h);dst=render_scene(model,f['camera'],w,h)
    t=lambda ru,en:en if lang=='en' else ru
    fig,axs=plt.subplots(1,3,figsize=(9,3),layout='constrained')
    for ax,img,label in zip(axs,[source,dst,np.mean(abs(source-dst),axis=2)],
                            [t('До упаковки','Before packing'),f'{precision} · {len(packet)} B',t('Ошибка RGB','RGB error')]):
        if img.ndim==2: ax.imshow(img,vmin=0,vmax=.2,cmap='magma')
        else:ax.imshow(np.clip(img,0,1))
        ax.set_title(label);ax.axis('off')
    fig.suptitle(f"{view} · {len(model['splats'])}/117 · MSE {scores(dst,source)['mse']:.7f}")
    return fig


def motion_figure(lang='en',mode='keys',stride=8,quantum=.002,time=.37,angle=32):
    t=lambda ru,en:en if lang=='en' else ru
    model,xyz=_motion();packet=pack_motion(xyz,mode,stride,quantum)
    track=unpack_motion(packet);dst=pose_with_centers(model['canonical'],sample_track(track,time))
    cam=orbit_camera(angle,55);src=scene_at(model,time)
    fig,axs=plt.subplots(1,3,figsize=(9,3),layout='constrained')
    for ax,scene,name in zip(axs[:2],[src,dst],[t('До упаковки','Before packing'),t('Из пакета','From packet')]):
        ax.imshow(render_scene(scene,cam,65,49));ax.axis('off');ax.set_title(name)
    ts=np.linspace(0,1,81)
    idx=int(np.argmax(np.ptp(xyz[:,:,0],axis=0)))
    axs[2].plot(ts,[scene_at(model,float(v))['splats'][idx]['center'][0] for v in ts],label=t('исходная','source'))
    axs[2].plot(ts,[sample_track(track,v)[idx,0] for v in ts],'--',label=t('из пакета','decoded'))
    axs[2].axvline(time,color='.5',lw=.9);axs[2].set(xlabel=t('время, с','time, s'),ylabel='x, m')
    axs[2].legend(fontsize=8);fig.suptitle(f'{mode} · {len(packet)} B · t = {time:.2f} s · {angle}°')
    return fig


def study_figure(report,lang='en',max_bytes=None,min_psnr=None,priority=None):
    t=lambda ru,en:en if lang=='en' else ru
    cfg=dict(report['settings'])
    if max_bytes is not None:cfg['maxBytes']=max_bytes
    if min_psnr is not None:cfg['minPsnr']=min_psnr
    if priority is not None:cfg['priority']=priority
    rows=report['candidates'];winner=choose_candidate(rows,cfg)
    fig,(ax,camera)=plt.subplots(1,2,figsize=(8.3,3.2),layout='constrained')
    for i,r in enumerate(rows):
        ax.scatter(r['bytes'],r['validationPsnr'],s=55,color='#687a3e' if r is winner else '#637cc7')
        ax.annotate(str(i+1),(r['bytes'],r['validationPsnr']),xytext=(3,3),textcoords='offset points')
    ax.axhline(cfg['minPsnr'],ls='--',color='#b26730');ax.axvline(cfg['maxBytes'],ls='--',color='#b26730')
    ax.set(xlabel=t('Размер, байт','Size, bytes'),ylabel='PSNR (view-00), dB')
    if winner:
        scene,frames=_photos();f=frames['view-04.jpg']
        _,precision,keep,color=next(c for c in static_candidates() if c[0]==winner['name'])
        dst=render_scene(unpack_static(pack_static(scene,precision,keep,color)),f['camera'],80,60)
        camera.imshow(dst);camera.set_title(f"{winner['name']} · {scores(dst,f['rgb'])['psnr']:.2f} dB")
    else:camera.text(.5,.5,t('Нет подходящего пакета','No eligible packet'),ha='center',va='center')
    camera.axis('off');fig.suptitle(t('Выбор по view-00, проверка на view-04','Selection on view-00, check on view-04'))
    return fig


def codec_lab(lesson,lang='en'):
    t=lambda ru,en:en if lang=='en' else ru
    def slider(label,lo,hi,value,step=1):
        cls=widgets.IntSlider if isinstance(step,int) else widgets.FloatSlider
        return cls(description=label,min=lo,max=hi,value=value,step=step,continuous_update=False,
                   style={'description_width':'initial'})
    def choice(label,opts,value):
        return widgets.Dropdown(description=label,options=opts,value=value,style={'description_width':'initial'})
    if lesson==27:
        controls=dict(count=slider(t('Пятна','Splat count'),1,1000,117),degree=slider(t('Порядок','Degree'),0,3,0),
                      width=slider(t('Ширина, px','Width, px'),4,320,80));draw=storage_figure
    elif lesson==28:
        controls=dict(keep=slider(t('Сохранено, %','Retained, %'),1,100,100),
                      precision=choice(t('Числа','Numbers'),['u8','f32'],'u8'),
                      color=choice(t('Цвет','Color'),['rgb','gray'],'rgb'),
                      view=choice(t('Камера','Camera'),['view-00.jpg','view-04.jpg','view-08.jpg'],'view-04.jpg'));draw=static_figure
    elif lesson==29:
        controls=dict(mode=choice(t('Способ','Method'),['dense','keys','residual'],'keys'),stride=slider(t('Шаг опор','Anchor stride'),1,16,8),
                      quantum=slider(t('Цена кода, м','Residual unit, m'),.001,.03,.002,.001),
                      time=slider(t('Время, с','Time, s'),0.,1.,.37,.01),angle=slider(t('Ракурс, °','View, °'),-45,45,32));draw=motion_figure
    else:
        report=benchmark_photo()
        controls=dict(max_bytes=slider(t('Лимит, байт','Byte limit'),1,10000,2200),
                      min_psnr=slider(t('Порог PSNR','PSNR floor'),0.,60.,16.,.1),
                      priority=choice(t('Приоритет','Priority'),['quality','size','latency'],'quality'))
        draw=lambda lang,**kwargs:study_figure(report,lang,**kwargs)
    def update(**kwargs):
        fig=draw(lang=lang,**kwargs);plt.show();plt.close(fig)
    out=widgets.interactive_output(update,controls)
    return widgets.VBox([*controls.values(),out])
