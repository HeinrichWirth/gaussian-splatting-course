"""Differentiable CPU refinement of a calibrated, diffuse 3D Gaussian scene."""
from copy import deepcopy
import numpy as np
import torch


def refine_scene(scene, frames, steps=20, rate=.015, pixels_per_step=None, seed=0):
    """Fit geometry and appearance. Optionally sample one training view per update.

    By default history contains the full loss at each step. In sampled mode it
    contains the minibatch losses followed by a final full training loss.
    """
    if steps<0 or not np.isfinite(rate) or rate<0:
        raise ValueError("Steps and rate must be finite and nonnegative")
    if pixels_per_step is not None and (not isinstance(pixels_per_step, int) or pixels_per_step < 1):
        raise ValueError("The sample size must be a positive integer")
    rng = np.random.default_rng(seed)
    views=[f for f in frames if f["split"]=="train"]
    if not views:
        raise ValueError("No train views")
    splats=scene["splats"]
    def tensor(values):
        return torch.tensor(np.asarray(values),dtype=torch.float64)
    def parameter(values):
        return torch.nn.Parameter(values)
    def logits(values):
        return torch.logit(tensor(values),eps=1e-5)
    center=parameter(tensor([p["center"] for p in splats]))
    log_s=parameter(tensor([p["scales"] for p in splats]).log())
    quat=parameter(tensor([p["rotation"] for p in splats]))
    color=parameter(logits([p["color"] for p in splats]))
    opacity=parameter(logits([p["opacity"] for p in splats]))
    optimizer=torch.optim.Adam([center,log_s,quat,color,opacity],lr=rate)
    def render(frame, sample_size=None):
        target=tensor(frame["rgb"])
        h,w=target.shape[:2]
        camera=frame["camera"]
        R=tensor(camera["rotation"])
        xyz=(center-tensor(camera["center"]))@R
        z=xyz[:,2].clamp_min(camera.get("near",.05))
        f=np.broadcast_to(camera["focal"],(2,))
        fx,fy=float(f[0]),float(f[1])
        cx,cy=camera.get("principal",[(w-1)/2,(h-1)/2])
        uv=torch.stack([fx*xyz[:,0]/z+cx,fy*xyz[:,1]/z+cy],dim=1)
        zero=torch.zeros_like(z)
        J=torch.stack([fx/z,zero,-fx*xyz[:,0]/z.square(),zero,fy/z,
                       -fy*xyz[:,1]/z.square()],dim=1).reshape(-1,2,3)
        qw,qx,qy,qz=torch.nn.functional.normalize(quat,dim=1).unbind(1)
        rot=torch.stack([
            1-2*(qy*qy+qz*qz),2*(qx*qy-qw*qz),2*(qx*qz+qw*qy),
            2*(qx*qy+qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz-qw*qx),
            2*(qx*qz-qw*qy),2*(qy*qz+qw*qx),1-2*(qx*qx+qy*qy)
        ],dim=1).reshape(-1,3,3)
        transform=J@R.T@rot*log_s.exp().unsqueeze(1)
        cov=transform@transform.transpose(1,2)
        if sample_size is not None and sample_size < h*w:
            indices = torch.from_numpy(rng.integers(h*w, size=sample_size))
            yy,xx = (indices // w)[:,None], (indices % w)[:,None]
            target = target[yy,xx]
        else:
            yy,xx=torch.meshgrid(torch.arange(h,dtype=z.dtype),torch.arange(w,dtype=z.dtype),indexing="ij")
        delta=torch.stack([xx[None]-uv[:,0,None,None],yy[None]-uv[:,1,None,None]],dim=-1)
        precision=torch.linalg.inv(cov)
        distance=torch.einsum("nhwi,nij,nhwj->nhw",delta,precision,delta)
        alpha=opacity.sigmoid()[:,None,None]*torch.exp(-.5*distance)
        alpha=alpha*(xyz[:,2]>camera.get("near",.05))[:,None,None]
        order=xyz[:,2].detach().argsort(stable=True)
        a=alpha[order]
        through=torch.cumprod(torch.cat([torch.ones_like(a[:1]),1-a]),dim=0)
        rgb=torch.einsum("nhw,nc->hwc",through[:-1]*a,color.sigmoid()[order])+through[-1,:,:,None]
        return (rgb-target).square().mean()
    history=[]
    for i in range(steps+1):
        if pixels_per_step is not None and i < steps:
            loss = render(views[rng.integers(len(views))], pixels_per_step)
        else:
            # Evaluating the final full images one at a time keeps the graph small.
            with torch.set_grad_enabled(i < steps):
                loss=torch.stack([render(f) for f in views]).mean()
        history.append(float(loss.detach()))
        if i<steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model=deepcopy(scene)
    with torch.no_grad():
        values=[center,log_s.exp(),torch.nn.functional.normalize(quat,dim=1),color.sigmoid(),opacity.sigmoid()]
        for i,p in enumerate(model["splats"]):
            for key,arr in zip(("center","scales","rotation","color","opacity"),values):
                p[key]=arr[i].tolist()
    return model,history
