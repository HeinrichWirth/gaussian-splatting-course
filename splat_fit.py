"""Differentiable two-splat exercise. Cameras translate along X with parallel axes."""
import torch

from splat_render import make_scene, render_scene, splat, translated_camera


def render_parallel(centers, scales, colors, opacity, baseline, size=33, focal=28):
    """Axis-aligned 3D covariances, the same local perspective and white background."""
    device, dtype = centers.device, centers.dtype
    x, y, z = (centers-centers.new_tensor([baseline, 0., 0.])).unbind(1)
    sx, sy, sz = scales.unbind(1)
    u = focal*x/z+(size-1)/2
    v = focal*y/z+(size-1)/2
    a = (focal*sx/z)**2+(focal*x*sz/z**2)**2
    d = (focal*sy/z)**2+(focal*y*sz/z**2)**2
    b = focal**2*x*y*sz**2/z**4
    grid = torch.arange(size, dtype=dtype, device=device)
    row, col = torch.meshgrid(grid, grid, indexing="ij")
    dx, dy = col-u[:, None, None], row-v[:, None, None]
    det = (a*d-b*b)[:, None, None]
    q = (d[:, None, None]*dx**2 - 2*b[:, None, None]*dx*dy+a[:, None, None]*dy**2)/det
    alpha = opacity[:, None, None]*torch.exp(-q/2)
    remaining = torch.ones_like(row)
    rgb = torch.zeros((size, size, 3), dtype=dtype, device=device)
    for i in torch.argsort(z, stable=True):
        rgb = rgb + (remaining*alpha[i])[..., None]*colors[i]
        remaining = remaining*(1-alpha[i])
    return rgb + remaining[..., None]


def train_two_splats(steps=45):
    """Recover centers and RGB from two calibrated frames; a third stays held out."""
    scene = make_scene(splat([-.42,-.14,3.55],[.34,.24,.18],[.9,.22,.10],.85),
                       splat([.5,.18,4.6],[.26,.44,.22],[.14,.27,.88],.80))
    baselines = (-.8, .65, .12)
    targets = [torch.as_tensor(render_scene(scene, translated_camera(x), 33, 33))
               for x in baselines]
    centers = torch.tensor([[-.20,-.05,3.05],[.3,.04,5.2]],dtype=torch.float64,requires_grad=True)
    colors = torch.tensor([[.65,.45,.22],[.3,.4,.65]],dtype=torch.float64,requires_grad=True)
    scales = centers.new_tensor([s["scales"] for s in scene["splats"]])
    opacity = centers.new_tensor([s["opacity"] for s in scene["splats"]])
    before, history = None, []
    optimizer = torch.optim.Adam([centers,colors],lr=.04)
    for iteration in range(steps+1):
        frames = [render_parallel(centers,scales,colors,opacity,x) for x in baselines]
        losses = [(pred-target).square().mean() for pred,target in zip(frames,targets)]
        train = (losses[0]+losses[1])/2
        history.append((float(train.detach()),float(losses[2].detach())))
        if before is None:
            before = frames[-1].detach().numpy()
        if iteration == steps:
            break
        optimizer.zero_grad()
        train.backward()
        optimizer.step()
        with torch.no_grad():
            colors.clamp_(0,1)
            centers[:,2].clamp_(1.25,7)
    fitted = make_scene(*(splat(center.tolist(),s["scales"],color.tolist(),s["opacity"])
                          for center,color,s in zip(centers.detach(),colors.detach(),scene["splats"])))
    return dict(scene=fitted, history=history, before=before, after=frames[-1].detach().numpy(),
                target=targets[-1].numpy())
