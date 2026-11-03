"""Small binary formats for a static RGB scene and fixed-identity center tracks."""
import json
from copy import deepcopy
from pathlib import Path
from struct import Struct

import numpy as np

from reconstruction import benchmark, load_colmap, scores
from splat_render import render_scene

ASSETS = Path(__file__).resolve().parent / 'site' / 'dist' / 'assets' / 'reconstruction'
STATIC = Struct('<4sBBH')
MOTION = Struct('<4sBBHHHf')
MAX_TRACK_POINTS = 250_000
DEFAULT_STUDY = dict(format='gs-course-codec-study-v1', width=80, height=60, repeats=3,
                     maxBytes=2200, minPsnr=16, priority='quality')


def memory_budget(count=117, degree=0, width=80, height=60):
    """A specified float32 layout. These subtotals exclude allocator and renderer scratch."""
    if min(count, degree, width, height) < 0 or min(width, height) == 0:
        raise ValueError('Invalid dimensions or basis degree')
    channels = 3 * (degree + 1)**2
    fields = dict(center=count*12, scales=count*12, rotation=count*16,
                  opacity=count*4, color=count*channels*4)
    weights = sum(fields.values())
    return dict(fields=fields, weights=weights, rgb=width*height*3*4,
                viewer=weights + width*height*3*4, adam=4*weights)


def photo_data():
    data = Path(__file__).resolve().parent / 'data' / 'photographic-example'
    frames, _ = load_colmap(data/'sparse'/'0', data/'images', max_side=80, test_names=['view-04.jpg'])
    return json.loads((ASSETS/'photo-reference.json').read_text('utf8')), {f['name']: f for f in frames}


def pack_static(scene, precision='u8', keep=100, color='rgb'):
    """GSC1, stable row order. Ranking uses opacity times the two largest scales."""
    if (scene.get('format') != 'gs-course-3d-v1' or precision not in ('f32','u8')
            or color not in ('rgb','gray') or not 0 < keep <= 100):
        raise ValueError('Unsupported static codec settings')
    points = scene['splats']
    if not 1 <= len(points) <= 65535 or any('sh' in p for p in points):
        raise ValueError('Expected 1..65535 constant-color splats')
    count = max(1, int(np.ceil(len(points)*keep/100)))
    rank = sorted(range(len(points)), key=lambda i: -points[i]['opacity'] * np.prod(sorted(points[i]['scales'])[-2:]))
    channels = 3 if color == 'rgb' else 1
    rows = []
    for i in sorted(rank[:count]):
        p = points[i]
        rgb = p['color'] if channels == 3 else [sum(p['color'])/3]
        rows.append([*p['center'], *p['scales'], *p['rotation'], p['opacity'], *rgb])
    rows = np.array(rows, dtype=np.float64)
    if not (np.isfinite(rows).all() and np.all(rows[:, 3:6] > 0)
            and np.all(np.linalg.norm(rows[:, 6:10], axis=1) > 0)
            and np.all((rows[:, 10:] >= 0) & (rows[:, 10:] <= 1))):
        raise ValueError('Invalid Gaussian parameters')
    header = STATIC.pack(b'GSC1', 4 if precision == 'f32' else 1, channels, count)
    if precision == 'f32':
        return header + rows.astype('<f4').tobytes()
    # Quaternion components are bounded after normalization. Center and scale ranges
    # are stored as float32 and used by both encoder and decoder.
    rows[:, 6:10] /= np.linalg.norm(rows[:, 6:10], axis=1)[:, None]
    bounds = np.stack([rows[:, :6].min(axis=0), rows[:, :6].max(axis=0)]).astype('<f4')
    lo = np.r_[bounds[0].astype(float), [-1.]*4, [0.]*(1+channels)]
    hi = np.r_[bounds[1].astype(float), [1.]*4, [1.]*(1+channels)]
    unit = np.divide(rows-lo, hi-lo, out=np.zeros_like(rows), where=hi!=lo)
    codes = np.floor(np.clip(unit,0,1)*255+.5).astype('u1')
    return header + bounds.tobytes() + codes.tobytes()


def unpack_static(packet):
    if len(packet) < STATIC.size:
        raise ValueError('Truncated GSC1 header')
    magic, size, colors, count = STATIC.unpack_from(packet)
    columns = 11 + colors
    if (magic != b'GSC1' or size not in (1,4) or colors not in (1,3) or not count
            or len(packet) != 8+(48 if size == 1 else 0)+count*columns*size):
        raise ValueError('Invalid GSC1 packet')
    if size == 4:
        rows = np.frombuffer(packet, '<f4', offset=8).reshape(count, columns)
    else:
        bounds = np.frombuffer(packet, '<f4', count=12, offset=8).astype(float).reshape(2,6)
        if not np.isfinite(bounds).all() or np.any(bounds[1] < bounds[0]):
            raise ValueError('Invalid geometry bounds')
        lo = np.r_[bounds[0], [-1.]*4, [0.]*(1+colors)]
        hi = np.r_[bounds[1], [1.]*4, [1.]*(1+colors)]
        rows = lo + np.frombuffer(packet, 'u1', offset=56).reshape(count, columns)/255*(hi-lo)
    if (not np.isfinite(rows).all() or np.any(rows[:,3:6]<=0)
            or np.any(np.linalg.norm(rows[:,6:10],axis=1)==0)
            or np.any((rows[:,10:]<0) | (rows[:,10:]>1))):
        raise ValueError('Invalid decoded Gaussian')
    splats = []
    for row in rows.tolist():
        rgb = row[11:] if colors == 3 else [row[11]]*3
        splats.append(dict(center=row[:3], scales=row[3:6], rotation=row[6:10], opacity=row[10], color=rgb))
    return {'format':'gs-course-3d-v1','splats':splats}


def anchor_frames(frames, stride):
    return sorted({*range(0,frames,stride),frames-1})


def _interpolate(keys, values, frame):
    j = max(0, min(len(keys)-2, int(np.searchsorted(keys, frame, side='right'))-1))
    u = (frame-keys[j])/(keys[j+1]-keys[j])
    return (1-u)*values[j] + u*values[j+1]


def pack_motion(centers, mode='residual', stride=4, quantum=.002):
    """GSM1 stores uniformly timed XYZ tracks. Pose-independent appearance is external."""
    xyz = np.asarray(centers, dtype=float)
    if (xyz.ndim != 3 or xyz.shape[2] != 3 or not 2 <= len(xyz) <= 65535
            or not 1 <= xyz.shape[1] <= 65535 or len(xyz)*xyz.shape[1] > MAX_TRACK_POINTS or not np.isfinite(xyz).all()
            or int(stride) != stride or not 1 <= stride <= 65535
            or not np.isfinite(quantum) or quantum <= 0):
        raise ValueError('Invalid motion dimensions, stride or quantum')
    kinds = ('dense', 'keys', 'residual')
    code = kinds.index(mode)
    stride = 1 if mode == 'dense' else int(stride)
    quantum = float(np.float32(quantum))
    keys = anchor_frames(len(xyz),stride)
    values = xyz[keys].astype('<f4')
    packet = MOTION.pack(b'GSM1',code,0,len(xyz),xyz.shape[1],stride,quantum) + values.tobytes()
    if mode == 'residual':
        others = sorted(set(range(len(xyz)))-set(keys))
        if others:
            pred = np.stack([_interpolate(keys, values.astype(float), i) for i in others])
            units = (xyz[others]-pred)/quantum
            packet += np.clip(np.floor(units+.5), -127,127).astype('i1').tobytes()
    return packet


def unpack_motion(packet):
    if len(packet) < MOTION.size:
        raise ValueError('Truncated GSM1 header')
    magic, code, reserved, frames, count, stride, quantum = MOTION.unpack_from(packet)
    if (magic != b'GSM1' or code not in (0,1,2) or reserved or min(frames-1,count,stride)<1
            or frames*count > MAX_TRACK_POINTS or (code == 0 and stride != 1)
            or not np.isfinite(quantum) or quantum <= 0):
        raise ValueError('Invalid GSM1 header')
    keys = anchor_frames(frames,stride)
    used = 16+len(keys)*count*12
    if len(packet) != used + ((frames-len(keys))*count*3 if code==2 else 0):
        raise ValueError('Invalid GSM1 length')
    values = np.frombuffer(packet, '<f4', offset=16, count=len(keys)*count*3).astype(float).reshape(-1,count,3)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite motion key')
    if code==0:
        return values
    decoded = np.stack([_interpolate(keys,values,i) for i in range(frames)])
    if code==2:
        others = sorted(set(range(frames))-set(keys))
        decoded[others] += np.frombuffer(packet, 'i1', offset=used).reshape(len(others),count,3)*quantum
    return decoded


def sample_track(decoded, time):
    if not 0 <= time <= 1:
        raise ValueError('Time must be in [0, 1]')
    f = time*(len(decoded)-1)
    low = min(int(f),len(decoded)-2)
    u = f-low
    return decoded[low]*(1-u) + decoded[low+1]*u


def motion_data(frames=17, model=None):
    from deformation import example_model, scene_at
    model = example_model() if model is None else model
    centers = [[s['center'] for s in scene_at(model,float(t))['splats']]
               for t in np.linspace(0,model['duration'],frames)]
    return model, np.array(centers)


def pose_with_centers(canonical, centers):
    scene = deepcopy(canonical)
    for point, xyz in zip(scene['splats'], centers, strict=True):
        point['center'] = list(xyz)
    return scene


def static_candidates():
    return [('f32-rgb', 'f32', 100, 'rgb'), ('u8-rgb', 'u8', 100, 'rgb'),
            ('u8-half', 'u8', 50, 'rgb'), ('u8-gray-half', 'u8', 50, 'gray')]


def study_settings(settings=None, repeats=None):
    cfg = dict(DEFAULT_STUDY if settings is None else settings)
    if repeats is not None:
        cfg['repeats'] = repeats
    if (set(cfg)!=set(DEFAULT_STUDY) or cfg['format']!=DEFAULT_STUDY['format']
            or (cfg['width'], cfg['height'])!=(80,60) or type(cfg['repeats']) is not int
            or not 1 <= cfg['repeats'] <= 15 or type(cfg['maxBytes']) is not int
            or not 1 <= cfg['maxBytes'] <= 100000 or not 0 <= cfg['minPsnr'] <= 60
            or cfg['priority'] not in ('quality','size','latency')):
        raise ValueError('Unsupported experiment settings')
    return cfg


def choose_candidate(rows, cfg):
    eligible = [r for r in rows if r['bytes'] <= cfg['maxBytes'] and r['validationPsnr'] >= cfg['minPsnr']]
    order = dict(quality=lambda r:-r['validationPsnr'],size=lambda r:r['bytes'],
                 latency=lambda r:r['renderMs'])
    return min(eligible,key=order[cfg['priority']],default=None)


def benchmark_photo(settings=None, repeats=None):
    cfg = study_settings(settings,repeats)
    scene, frames = photo_data()
    train = frames['view-00.jpg']
    rows = []
    packets = {}
    for name, precision, keep, color in static_candidates():
        packet = pack_static(scene,precision,keep,color)
        packets[name] = packet
        decoded = unpack_static(packet)
        def render():
            return render_scene(decoded,train['camera'],cfg['width'],cfg['height'])
        rows.append(dict(name=name, bytes=len(packet),splats=len(decoded['splats']),
                         validationPsnr=scores(render(),train['rgb'])['psnr'],
                         decodeMs=benchmark(lambda:unpack_static(packet),cfg['repeats']),
                         renderMs=benchmark(render,cfg['repeats']),
                         bufferBytes=cfg['width']*cfg['height']*3*8))
    winner = choose_candidate(rows,cfg)
    held = None
    if winner:
        test = frames['view-04.jpg']
        result = scores(render_scene(unpack_static(packets[winner['name']]),test['camera'],80,60),test['rgb'])
        held = dict(view=test['name'],mse=result['mse'],psnr=result['psnr'])
    return dict(settings=cfg,candidates=rows,selected=winner['name'] if winner else None,
                validationView=train['name'],test=held)
