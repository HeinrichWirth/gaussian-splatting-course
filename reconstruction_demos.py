"""Notebook figures for the rasterizer, two-camera capture, fitting and evaluation."""
from pathlib import Path
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from reconstruction import (small_scene,orbit,finite_render,triangulate,load_colmap,
                            synthetic_capture,fit_colors,evaluate,scores,render_scene)

ROOT=Path(__file__).resolve().parent


def photo_capture(max_side=40, max_points=24):
    data=ROOT/"data"/"photographic-example"
    return load_colmap(data/"sparse"/"0",data/"images",max_side=max_side,max_points=max_points,
                       test_names=["view-04.jpg"])


def label(lang,ru,en):
    return ru if lang=="ru" else en


def tiles_figure(tile=8,cutoff=3.,angle=0,lang="ru"):
    w,h=73,57
    sc=small_scene();cam=orbit(angle,w,h)
    full=render_scene(sc,cam,w,h)
    img,stats=finite_render(sc,cam,w,h,tile,cutoff)
    fig,ax=plt.subplots(1,3,figsize=(9,3.15),layout="constrained")
    ax[0].imshow(img);ax[0].set_title(label(lang,"Кадр","Frame"))
    ax[1].imshow(stats["tiles"],cmap="Blues",vmin=0,vmax=len(sc["splats"]),interpolation="nearest")
    ax[1].set_title(label(lang,"Пятен в тайле","Splats per tile"))
    delta=np.abs(img-full).mean(axis=2)
    im=ax[2].imshow(delta,cmap="magma",vmin=0,vmax=max(.007,delta.max()))
    ax[2].set_title(label(lang,"Отсечённые хвосты","Truncated tails"));fig.colorbar(im,ax=ax[2],shrink=.65)
    for a in ax:a.axis("off")
    return fig


def rays_figure(baseline=.4,noise=0.,depth=4.,lang="ru"):
    f=80;left=40;right=40-f*baseline/depth+noise;estimate=triangulate(f,baseline,left,right)
    fig,ax=plt.subplots(1,2,figsize=(8,3.1),layout="constrained")
    z=np.linspace(0,10)
    ax[0].plot(0*z,z,label="L");ax[0].plot(baseline+(right-40)/f*z,z,label="R")
    ax[0].scatter([0],[depth],s=75,marker="x",color="green",label="Z")
    if estimate is not None and estimate<=10:ax[0].scatter([0],[estimate],facecolors="none",edgecolors="purple",s=100,label="Z′")
    ax[0].scatter([0,baseline],[0,0],color="black",marker="^");ax[0].legend(fontsize=7);ax[0].set(xlabel="X (m)",ylabel="Z (m)",xlim=(-1.1,1.1),ylim=(-.4,10))
    ds=np.linspace(.5,35,180);ax[1].plot(ds,f*baseline/ds,color="#4b69ac")
    if estimate is not None:ax[1].scatter([left-right],[estimate],color="#cb743b")
    ax[1].set(xlabel=label(lang,"Диспаратность, пикс.","Disparity, px"),ylabel="Z (m)",ylim=(0,12))
    for a in ax:a.grid(alpha=.2)
    return fig


def view_figure(frames,scene,index=1,offset=0,lang="ru"):
    f=frames[index];h,w=f["rgb"].shape[:2];cam=deepcopy(f["camera"])
    center=np.mean([p["center"] for p in scene["splats"]],axis=0)
    cam["center"]=(np.asarray(cam["center"])+offset*np.linalg.norm(center-cam["center"])*
                   np.asarray(cam["rotation"])[:,0]).tolist()
    im=render_scene(scene,cam,w,h)
    fig,ax=plt.subplots(1,2,figsize=(7.5,3.1),layout="constrained")
    ax[0].imshow(f["rgb"]);ax[0].set_title(f["name"]+" · "+f["split"])
    ax[1].imshow(np.clip(im,0,1));ax[1].set_title(label(lang,"Модель","Model")+f" · Δ={offset:.2f}")
    for a in ax:a.axis("off")
    return fig


def damage(scene,kind,strength):
    model=deepcopy(scene)
    if kind=="blur":
        for p in model["splats"]:p["scales"]=(np.array(p["scales"])*(1+strength)).tolist()
    if kind=="tint":
        for p in model["splats"]:p["color"][1]*=1-strength
    if kind=="floater" and strength>0:
        p=deepcopy(model["splats"][2]);p.update(center=[.14,-.1,2.55],scales=[.14,.26,.12],color=[.12,.18,.18],opacity=strength)
        model["splats"].append(p)
    return model


def error_figure(frames,scene,index=1,kind="blur",strength=.45,lang="ru"):
    f=frames[index];h,w=f["rgb"].shape[:2]
    model=damage(scene,kind,strength);img=render_scene(model,f["camera"],w,h)
    fig,ax=plt.subplots(1,3,figsize=(8.5,3.0),layout="constrained")
    ax[0].imshow(f["rgb"]);ax[0].set_title(label(lang,"Эталон","Reference"))
    ax[1].imshow(img);ax[1].set_title(label(lang,"Модель","Model"))
    im=ax[2].imshow(np.abs(img-f["rgb"]).mean(axis=2),cmap="gray",vmin=0,vmax=1)
    ax[2].set_title(label(lang,"Среднее |RGB|","Mean |RGB error|"));fig.colorbar(im,ax=ax[2],shrink=.7)
    for a in ax:a.axis("off")
    return fig


def controls(plot,**ranges):
    """A saved baseline figure and live controls share the same plotting function."""
    items={}
    for name,params in ranges.items():
        if isinstance(params,list):
            items[name]=widgets.Dropdown(options=params,description=name)
        else:
            low,high,step,value=params
            items[name]=widgets.FloatSlider(min=low,max=high,step=step,value=value,description=name,continuous_update=False)
    def show(**kwargs):
        fig=plot(**kwargs)
        display(fig)
        plt.close(fig)
    output=widgets.interactive_output(show,items)
    display(widgets.VBox([*items.values(),output]))
