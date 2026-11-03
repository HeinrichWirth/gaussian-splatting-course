"""Experiments on the calibrated sculpture capture in lessons 19–21."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reconstruction_demos import photo_capture
from splat_render import footprint_alpha, project_splats, render_scene

ROOT = Path(__file__).resolve().parent


def feature_track(point_id=148):
    """Join the image observations to a sparse point by ID, not by row number."""
    directory = ROOT / 'data/photographic-example/sparse/0'
    def records(name):
        return [line for line in (directory / name).read_text().splitlines()
                if not line.startswith('#')]
    points = (row.split() for row in records('points3D.txt') if row)
    point = next(row for row in points if int(row[0]) == point_id)
    camera = next(row.split() for row in records('cameras.txt') if row)
    rows = iter(records('images.txt'))
    observations = {}
    for header in rows:
        samples = np.fromstring(next(rows), sep=' ').reshape(-1, 3)
        for u, v, feature in samples:
            if feature == point_id:
                observations[header.split(maxsplit=9)[-1]] = [u, v]
    return dict(id=point_id, xyz=list(map(float, point[1:4])), uv=observations,
                width=int(camera[2]), height=int(camera[3]))


def measured_pixel(track, frame):
    h, w = frame['rgb'].shape[:2]
    return np.array(track['uv'][frame['name']]) * [w/track['width'], h/track['height']] - .5


def camera_ray(frame, pixel):
    cam = frame['camera']
    xy = (pixel - np.array(cam['principal'])) / cam['focal']
    direction = np.array(cam['rotation']) @ np.r_[xy, 1.]
    return np.array(cam['center']), direction / np.linalg.norm(direction)


def intersect_observations(frame_a, pixel_a, frame_b, pixel_b):
    a, r = camera_ray(frame_a, pixel_a)
    b, s = camera_ray(frame_b, pixel_b)
    lengths, _, rank, _ = np.linalg.lstsq(np.column_stack((r, -s)), b-a, rcond=None)
    if rank < 2 or min(lengths) <= 0:
        raise ValueError("The two forward rays do not constrain a visible point")
    p, q = a + lengths[0]*r, b + lengths[1]*s
    return (p+q)/2, np.linalg.norm(p-q)


def reproject(xyz, frame):
    cam = frame['camera']
    point = np.asarray(cam['rotation']).T @ (xyz - np.asarray(cam['center']))
    return np.asarray(cam['principal']) + cam['focal']*point[:2]/point[2]


def track_figure(frames, track, partner=8, offset=0., lang='ru'):
    names = ['view-03.jpg', f'view-{int(partner):02}.jpg', 'view-07.jpg']
    a,b,c = [next(f for f in frames if f['name']==name) for name in names]
    ua, ub, uc = [measured_pixel(track,f) for f in (a,b,c)]
    ub = ub + [offset, 0]
    point, _ = intersect_observations(a,ua,b,ub)
    fig, axes = plt.subplots(1,3,figsize=(9,3.4),layout='constrained')
    for ax, f, mark in zip(axes,(a,b,c),(ua,ub,uc)):
        ax.imshow(f['rgb'],interpolation='nearest')
        ax.scatter(*mark,s=90,marker='+',color='#ffae4a')
        pred = reproject(point,f)
        ax.scatter(*pred,s=140,facecolors='none',edgecolors='#4dbdff')
        ax.set(xlim=(mark[0]-6,mark[0]+6),ylim=(mark[1]+6,mark[1]-6),
               title=f['name'],xlabel='u, px',ylabel='v, px')
    err = np.linalg.norm(reproject(point,c)-uc)
    fig.suptitle(('Трек 148 · ошибка на третьем виде' if lang=='ru' else 'Track 148 · third-view error')+f' = {err:.3f} px')
    return fig


def pixel_ledger(scene, frame, u, v):
    h,w = frame['rgb'].shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        raise ValueError("Pixel outside the photograph")
    trans = 1.
    rows = []
    rgb = np.zeros(3)
    for spot in project_splats(scene,frame['camera'],w,h):
        alpha = float(footprint_alpha([u,v],spot))
        weight = trans*alpha
        rows.append((spot['index'], spot['depth'], alpha, trans, weight))
        rgb += weight*spot['color']
        trans *= 1-alpha
    return sorted(rows,key=lambda r:-r[-1]), trans, rgb+trans


def reference_photo():
    frames,_ = photo_capture(max_side=80)
    frame = next(f for f in frames if f['split']=='test')
    path = ROOT / 'site/dist/assets/reconstruction/photo-reference.json'
    return json.loads(path.read_text('utf8')), frame


def pixel_figure(scene,frame,u=40,v=30,lang='ru'):
    u,v=int(u),int(v)
    rows,background,pred = pixel_ledger(scene,frame,u,v)
    target=frame['rgb'][v,u]
    fig,axes=plt.subplots(1,3,figsize=(9.8,3.1),layout='constrained')
    axes[0].imshow(frame['rgb'])
    axes[0].scatter(u,v,color='orange',marker='+',s=90)
    axes[0].set_title(('Пиксель' if lang=='ru' else 'Pixel')+f' ({u}, {v})')
    axes[0].axis('off')
    names=[str(row[0]) for row in rows[:3]]+['BG']
    axes[1].bar(names,[r[-1] for r in rows[:3]]+[background],color='#607aa6')
    axes[1].set(title='T × α',ylim=(0,1),xlabel='ID / BG')
    x=np.arange(3)
    axes[2].bar(x-.18,target,.36,label='RGB',color='#84ab80')
    axes[2].bar(x+.18,pred,.36,label='GS',color='#d78e62')
    axes[2].set(xticks=x,xticklabels=list('RGB'),ylim=(0,1))
    axes[2].legend()
    return fig


def region_error(target,prediction,mask):
    sq = np.mean((prediction-target)**2,axis=2)
    return int(mask.sum()),float(sq[mask].mean()),float(sq[mask].sum()/sq.sum()) if sq.sum() else 0.


def region_figure(target,prediction,left=20,top=15,width=40,height=30,lang='ru'):
    mask=np.zeros(target.shape[:2],dtype=bool)
    mask[int(top):int(top+height),int(left):int(left+width)]=True
    count,mse,share=region_error(target,prediction,mask)
    h,w=mask.shape
    fig,axes=plt.subplots(1,3,figsize=(9.3,3.15),layout='constrained')
    delta=np.sqrt(np.mean((prediction-target)**2,axis=2))
    for ax,a,title in zip(axes,[target,prediction,delta],['RGB','GS','RMS Δ']):
        ax.imshow(a,cmap='magma',vmin=0,vmax=1)
        ax.add_patch(plt.Rectangle((left-.5,top-.5),width,height,fill=False,ec='#27d6df',lw=1.2))
        ax.set_title(title);ax.axis('off')
    fig.suptitle((f'Область: {count}/{h*w} пикс. · доля суммы квадратов' if lang=='ru'
                 else f'Region: {count}/{h*w} pixels · share of squared error')+f' {100*share:.1f}%')
    return fig
