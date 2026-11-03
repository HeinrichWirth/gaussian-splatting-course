"""Three small moving scenes. Lengths are metres, time is in seconds."""
import math

import numpy as np

from geometry3d import axis_angle_quaternion
from splat_render import make_scene, render_scene, splat, translated_camera


F = 80.
WIDTH, HEIGHT = 97, 61
U0 = (WIDTH - 1) / 2
DEPTH, BASELINE = 4., .8
ORANGE = [.93, .31, .09]
BLUE = [.16, .39, .74]


def horizontal_pixel(x, camera_x, z=DEPTH):
    return U0 + F * (x - camera_x) / z


def observing_pair(time=.8, object_speed=1., camera_speed=0., lag=0., gate=False):
    """Measure the moving marker twice; an optional opaque strip stands halfway to it."""
    views = []
    for exposure, baseline in [(time, 0.), (time + lag, BASELINE)]:
        camera_x = camera_speed * exposure + baseline
        x = object_speed * exposure
        u = horizontal_pixel(x, camera_x)
        ray_at_gate = (camera_x + x) / 2
        visible = 0 <= u < WIDTH and not (gate and .28 <= ray_at_gate <= .52)
        views.append(dict(time=exposure, camera_x=camera_x, x=x, u=u,
                          anchor=horizontal_pixel(-1.2, camera_x), visible=visible))
    disparity = views[0]['u'] - views[1]['u']
    # The usual synchronized estimate. A nonzero lag can invalidate its assumptions.
    estimate = F*BASELINE/disparity if all(v['visible'] for v in views) and disparity > 1e-9 else None
    return views, disparity, estimate


def observation_image(view, gate=False):
    scene = make_scene(splat([view['x'], 0, DEPTH], [.18, .14, .09], ORANGE, .96),
                       splat([-1.2, .54, DEPTH], [.14, .14, .10], BLUE, .94))
    image = render_scene(scene, translated_camera(view['camera_x'], F), WIDTH, HEIGHT)
    if gate:
        left, right = [horizontal_pixel(x, view['camera_x'], 2) for x in (.28, .52)]
        cols = np.arange(WIDTH)
        image[:, (cols >= left) & (cols <= right)] = [.23, .29, .30]
    return image


def true_x(time, bend=.2):
    return -.5 + .6*time + bend*time*(time-2)


def fit_path(time, middle=False, noise=0., bend=.2):
    """Fit the endpoints, optionally using a third measured position at one second."""
    start, end = true_x(0, bend), true_x(2, bend)
    line = start + (end-start)*time/2
    if not middle:
        return line
    offset = true_x(1, bend) + noise - (start+end)/2
    return line + offset*time*(2-time)


def trajectory_splat(time, center_x):
    scale_x = .24 + .08*time
    angle = 30*time
    rotation = axis_angle_quaternion([0, 0, 1], angle)
    return splat([center_x, 0, DEPTH], [scale_x, .11, .08], ORANGE, .96, rotation)


def persistent_pair(angle=90., noise=.4, strength=4., bend=0.):
    """Fit a neighbour between a 2D observation and a rotated rest edge."""
    if strength < 0:
        raise ValueError("the squared penalty needs a nonnegative weight")
    def edge(degrees):
        a = math.radians(degrees)
        return .8*np.array([math.cos(a), math.sin(a)])
    rigid, actual = edge(angle), edge(angle + bend)
    observed = actual + [noise, -noise/2]
    fitted = (observed + strength*rigid) / (1+strength)
    return dict(rigid=rigid, actual=actual, observed=observed, fitted=fitted,
                rigidity_error=np.linalg.norm(fitted-rigid),
                tracking_error=np.linalg.norm(fitted-actual))
