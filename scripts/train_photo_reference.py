"""Train the larger CPU example from the eight training photographs in lesson 20."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reconstruction import evaluate, load_colmap
from reconstruction_fit import refine_scene


def initial_scene(model):
    rows = [line.split() for line in (model / 'points3D.txt').read_text().splitlines()
            if line.strip() and not line.startswith('#')]
    xyz = np.array([r[1:4] for r in rows], dtype=float)
    rgb = np.array([r[4:7] for r in rows], dtype=float) / 255
    distances = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
    # Index 0 is the point itself. Its third neighbour sets a local length scale.
    widths = np.clip(np.partition(distances, 3, axis=1)[:, 3] * .65, .04, 1.2)
    splats = []
    for point, color, width in zip(xyz, rgb, widths):
        splats.append(dict(center=point.tolist(), scales=[width, width, width],
                           rotation=[1., 0., 0., 0.], color=color.tolist(), opacity=.72))
    return {'format': 'gs-course-3d-v1', 'splats': splats}


def scores(scene, frames):
    result = {}
    for split in ('train', 'test'):
        views = evaluate(scene, frames, split)
        result[split] = float(np.mean([view['mse'] for view in views]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--output', type=Path, default=Path('static-scene-result/reference.json'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    data = ROOT / 'data' / 'photographic-example'
    model = data / 'sparse' / '0'
    frames, _ = load_colmap(model, data/'images', max_side=80, test_names=['view-04.jpg'])
    scene = initial_scene(model)
    before = scores(scene, frames)
    trained, _ = refine_scene(scene, frames, steps=args.steps, rate=.009,
                              pixels_per_step=256, seed=17)
    after = scores(trained, frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trained, separators=(',', ':'), allow_nan=False), encoding='utf8')
    for split in ('train', 'test'):
        print(f'{split}: MSE {before[split]:.6f} -> {after[split]:.6f}, '
              f'PSNR {-10*np.log10(after[split]):.2f} dB')
    print(f'{len(trained["splats"])} splats -> {args.output}')


if __name__ == '__main__':
    main()
