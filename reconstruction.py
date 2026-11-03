"""Small multiview capture, finite-support rasterization and image-based fitting.

The camera convention and the scene file are shared with splat_render.py.
All pixel centers in the course renderer have integer coordinates.
"""
from copy import deepcopy
from pathlib import Path
from statistics import median
from time import perf_counter
import base64
import json
import math

import numpy as np
from PIL import Image
from geometry3d import quaternion_matrix
from splat_render import project_splats, render_scene


def small_scene():
    rows = [
        ([-.62, .13, 3.45], [.31, .45, .22], [.15, .45, .83]),
        ([.48, .18, 3.7], [.38, .41, .22], [.92, .31, .12]),
        ([.02, -.52, 3.28], [.31, .28, .20], [.95, .72, .12]),
        ([.02, .54, 3.04], [.25, .12, .19], [.18, .67, .36]),
        ([.38, -.38, 4.1], [.40, .12, .23], [.59, .23, .66]),
    ]
    return {"format": "gs-course-3d-v1", "splats": [
        dict(center=c, scales=s, rotation=[1, 0, 0, 0], color=rgb, opacity=.88)
        for c, s, rgb in rows]}


def orbit(angle, width=41, height=33):
    """Turn a camera around the scene center; positive angle moves it to the right."""
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    return dict(center=[3.5*s, 0., 3.5*(1-c)], rotation=[[c, 0, -s], [0, 1, 0], [s, 0, c]],
                focal=.97*width, principal=[(width-1)/2, (height-1)/2], near=.05)


def finite_render(scene, camera, width, height, tile=8, cutoff=3., tiled=True):
    """Tile list keeps the order from project_splats. Both paths truncate at q<=k²."""
    if min(width, height, tile) < 1 or not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Positive dimensions, tile size and cutoff required")
    projected = project_splats(scene, camera, width, height)
    nx, ny = math.ceil(width/tile), math.ceil(height/tile)
    buckets = [[[] for _ in range(nx)] for _ in range(ny)]
    prepared = []
    for index, p in enumerate(projected):
        cov, uv = np.asarray(p["covariance"]), np.asarray(p["mean"])
        prepared.append((uv, np.linalg.inv(cov), np.asarray(p["color"]), p["opacity"]))
        radius = cutoff * np.sqrt(np.diag(cov))
        first = np.maximum(0, np.ceil(uv-radius).astype(int) // tile)
        last = np.minimum([nx-1, ny-1], np.floor(uv+radius).astype(int) // tile)
        for ty in range(first[1], last[1]+1):
            for tx in range(first[0], last[0]+1):
                buckets[ty][tx].append(index)
    frame = np.ones((height, width, 3))
    pairs = 0
    for ty in range(ny):
        for tx in range(nx):
            y0, x0 = ty*tile, tx*tile
            y1, x1 = min(y0+tile,height), min(x0+tile,width)
            y, x = np.mgrid[y0:y1, x0:x1]
            ids = buckets[ty][tx] if tiled else range(len(prepared))
            pairs += x.size * len(ids)
            rgb, trans = np.zeros((*x.shape, 3)), np.ones(x.shape)
            for index in ids:
                uv, inv, color, opacity = prepared[index]
                dx, dy = x-uv[0], y-uv[1]
                q = inv[0,0]*dx**2 + 2*inv[0,1]*dx*dy + inv[1,1]*dy**2
                alpha = np.where(q <= cutoff**2, opacity*np.exp(-.5*q), 0.)
                rgb += (trans*alpha)[..., None]*color
                trans *= 1-alpha
            frame[y0:y1,x0:x1] = rgb + trans[...,None]
    return frame, {"pairs":pairs, "visible":len(projected),
                   "tiles":np.array([[len(b) for b in row] for row in buckets]),
                   "references":sum(len(b) for row in buckets for b in row)}


def triangulate(focal, baseline, u_left, u_right):
    """Rectified pair, identical intrinsics. None means no finite positive depth."""
    disparity = u_left-u_right
    return focal*baseline/disparity if baseline>0 and focal>0 and disparity>0 else None


def load_colmap(text_model, images_dir, max_side=48, max_points=32, holdout_every=4, test_names=None):
    """Read an undistorted text model. Pick training-visible seeds avoiding held-out RGB."""
    model, images_dir = Path(text_model), Path(images_dir).resolve()
    if min(max_side, max_points) < 1 or holdout_every < 2:
        raise ValueError("Invalid dataset limits")
    def lines(name):
        return (s for s in (model/name).read_text(encoding="utf8").splitlines()
                if not s.lstrip().startswith("#"))
    cameras = {}
    for line in lines("cameras.txt"):
        if not line.strip():
            continue
        ident, kind, w, h, *params = line.split()
        p = list(map(float, params))
        if kind=="SIMPLE_PINHOLE":
            f,cx,cy=p
            fx=fy=f
        elif kind=="PINHOLE":
            fx,fy,cx,cy=p
        else:
            raise ValueError("Expected undistorted PINHOLE or SIMPLE_PINHOLE")
        cameras[int(ident)] = int(w), int(h), fx,fy,cx,cy
    records = []
    iterator = iter(lines("images.txt"))
    for header in iterator:
        if not header.strip():
            continue
        fields=header.split(maxsplit=9)
        if len(fields)!=10:
            raise ValueError("Invalid COLMAP image header")
        image_id, camera_id, name=int(fields[0]), int(fields[8]), fields[9]
        rot=quaternion_matrix(list(map(float,fields[1:5])))
        center=-rot.T @ np.array(fields[5:8], dtype=float)
        track_line=next(iterator, None)
        if track_line is None:
            raise ValueError("Missing COLMAP observation row")
        observations=track_line.split()
        w,h,fx,fy,cx,cy=cameras[camera_id]
        scale=min(1,max_side/max(w,h))
        nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
        path=(images_dir/name).resolve()
        if not path.is_relative_to(images_dir):
            raise ValueError("Images must be inside the image directory")
        with Image.open(path) as source:
            if source.size!=(w,h):
                raise ValueError("Image and camera dimensions differ")
            rgb=np.array(source.convert("RGB").resize((nw,nh), Image.Resampling.LANCZOS))/255.
        cam=dict(center=center.tolist(), rotation=rot.T.tolist(), focal=[fx*nw/w,fy*nh/h],
                 principal=[cx*nw/w-.5,cy*nh/h-.5],near=.00001)
        records.append((name, image_id, cam, rgb, observations))
    records.sort(key=lambda r:r[0])
    if len(records)<2:
        raise ValueError("A training and a held-out view are required")
    if test_names is not None:
        test_names=set(test_names)
        names={r[0] for r in records}
        if not test_names or not test_names < names:
            raise ValueError("Held-out names must be a nonempty proper subset of image names")
    frames, seeds = [], {}
    for i,(name,image_id,cam,rgb,obs) in enumerate(records):
        held_out=(name in test_names if test_names is not None
                  else (i+1)%holdout_every==0 or i==len(records)-1)
        split="test" if held_out else "train"
        frames.append(dict(name=name, split=split, camera=cam, rgb=rgb))
        if split=="train":
            for pid in obs[2::3]:
                if int(pid)>=0:
                    seeds[int(pid)]=seeds.get(int(pid),0)+1
    points=[]
    for line in lines("points3D.txt"):
        parts=line.split()
        if parts and int(parts[0]) in seeds:
            points.append((int(parts[0]),list(map(float,parts[1:4]))))
    points.sort(key=lambda p:(-seeds[p[0]],p[0]))
    if not points:
        raise ValueError("No sparse points visible in a training frame")
    # A deterministic spread across the well-observed points.
    candidates=points[:max_points*8]
    indices=np.linspace(0,len(candidates)-1,min(max_points,len(candidates))).astype(int)
    xyz=np.array([candidates[i][1] for i in indices])
    radius=np.linalg.norm(np.ptp(xyz,axis=0))
    scale=max(radius/max(len(xyz)**(1/3)*4,1),1e-4)
    scene={"format":"gs-course-3d-v1","splats":[dict(center=p.tolist(),scales=[scale]*3,
           rotation=[1,0,0,0],opacity=.7,color=[.5]*3) for p in xyz]}
    return frames, scene


def synthetic_capture():
    scene=small_scene()
    frames=[]
    for angle,split in [(-32,"train"),(-15,"test"),(0,"train"),(21,"test"),(35,"train")]:
        cam=orbit(angle)
        rgb=np.rint(255*render_scene(scene,cam,41,33)).astype(np.uint8)/255.
        frames.append(dict(name=f"view_{angle:+03}.png",split=split,camera=cam,rgb=rgb))
    initial=deepcopy(scene)
    for p in initial["splats"]:
        p["color"]=[.5]*3
    return frames,initial


def write_capture(directory, frames, scene):
    """The folder stores actual PNG targets, camera poses, split, and initial model."""
    dest=Path(directory)
    dest.mkdir(parents=True,exist_ok=True)
    manifest={"format":"gs-course-capture-v1","initial":scene,"frames":[]}
    for i,frame in enumerate(frames):
        file=f"view-{i:03}.png"
        Image.fromarray(np.rint(255*np.clip(frame["rgb"],0,1)).astype("uint8")).save(dest/file)
        manifest["frames"].append(dict(name=frame["name"],file=file,split=frame["split"],camera=frame["camera"]))
    (dest/"capture.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")


def read_capture(directory):
    dest=Path(directory).resolve()
    m=json.loads((dest/"capture.json").read_text(encoding="utf8"))
    if m.get("format")!="gs-course-capture-v1":
        raise ValueError("Unknown capture format")
    frames=[]
    for frame in m["frames"]:
        p=(dest/frame["file"]).resolve()
        if not p.is_relative_to(dest) or frame["split"] not in ("train","test"):
            raise ValueError("Invalid frame")
        with Image.open(p) as im:
            frames.append({**frame,"rgb":np.asarray(im.convert("RGB"))/255.})
    if {f["split"] for f in frames}!={"train","test"}:
        raise ValueError("Both splits are required")
    return frames,m["initial"]


def color_basis(scene, frame):
    h,w=frame["rgb"].shape[:2]
    Y,X=np.mgrid[:h,:w]
    weights=np.zeros((h,w,len(scene["splats"])))
    T=np.ones((h,w))
    for p in project_splats(scene,frame["camera"],w,h):
        d=np.stack([X-p["mean"][0],Y-p["mean"][1]],axis=-1)
        q=np.einsum("...i,ij,...j->...",d,np.linalg.inv(p["covariance"]),d)
        a=p["opacity"]*np.exp(-.5*q)
        weights[:,:,p["index"]]=T*a
        T*=1-a
    return weights.reshape(h*w,-1),T.reshape(-1,1)


def fit_colors(scene, frames, steps=12):
    """Projected gradient descent; a row-sum bound sets a common safe step."""
    model=deepcopy(scene)
    train=[f for f in frames if f["split"]=="train"]
    if not train or steps<0:
        raise ValueError("A training view and nonnegative step count are required")
    matrices=[]; targets=[]
    for f in train:
        weights,background=color_basis(scene,f)
        matrices.append(weights)
        targets.append(f["rgb"].reshape(-1,3)-background)
    A,Y=np.concatenate(matrices),np.concatenate(targets)
    gram, rhs=A.T@A, A.T@Y
    scale=max(gram.sum(axis=1).max(),1e-15)
    colors=np.array([p["color"] for p in scene["splats"]],float)
    history=[float(np.mean((A@colors-Y)**2))]
    for _ in range(steps):
        colors=np.clip(colors-(gram@colors-rhs)/scale,0,1)
        history.append(float(np.mean((A@colors-Y)**2)))
    for p,c in zip(model["splats"],colors):
        p["color"]=c.tolist()
    return model,history


def scores(predicted, reference):
    p,t=np.asarray(predicted,float),np.asarray(reference,float)
    if p.shape!=t.shape or not p.size:
        raise ValueError("Images must have the same nonempty shape")
    error=p-t
    mse=float(np.mean(error**2))
    return dict(mae=float(np.abs(error).mean()),mse=mse,psnr=-10*math.log10(mse) if mse else math.inf)


def evaluate(scene,frames,split="test"):
    rows=[]
    for f in frames:
        if f["split"]==split:
            h,w=f["rgb"].shape[:2]
            img=render_scene(scene,f["camera"],w,h)
            rows.append(dict(name=f["name"],**scores(img,f["rgb"])))
    return rows


def benchmark(render, repeats=5):
    """Warm-up + median wall time; includes work inside the supplied callable only."""
    render()
    times=[]
    for _ in range(repeats):
        start=perf_counter()
        render()
        times.append((perf_counter()-start)*1000)
    return median(times)


def export_browser_capture(frames,initial,path):
    """Compact, single-file dataset for the browser lab. The bytes stay on this device."""
    packed=[]
    for f in frames:
        rgb=np.rint(np.clip(f["rgb"],0,1)*255).astype("uint8")
        packed.append(dict(name=f["name"],split=f["split"],camera=f["camera"],width=rgb.shape[1],
                           height=rgb.shape[0],rgb8=base64.b64encode(rgb.tobytes()).decode("ascii")))
    Path(path).write_text(json.dumps(dict(format="gs-course-capture-v1",initial=initial,frames=packed),
                                   separators=(",",":"),allow_nan=False),encoding="utf8")
