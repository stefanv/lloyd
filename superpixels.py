"""
Example of how to use Lloyd iteration to generate more-or-less
equally spaced points inside of a given boundary.
"""

from skimage import io
import numpy as np

from lloyd import lloyd
from moore import boundary_trace
from shapely.geometry import Polygon, Point

import matplotlib.pyplot as plt


N = 50
mask = io.imread('mask.png')[..., 0] > 50

rng = np.random.default_rng()

coords = np.vstack(np.where(mask)).T
boundary = boundary_trace(coords)
bpoly = Polygon(boundary)

# Sample N points within the polygon

bbox_min = np.min(boundary, axis=0) + 1e-3
bbox_max = np.max(boundary, axis=0)

sampled_points = []

while len(sampled_points) < N:
    r = rng.uniform(bbox_min[0], bbox_max[0])
    c = rng.uniform(bbox_min[1], bbox_max[1])

    x, y = c, r
    if bpoly.contains(Point(r, c)):
        sampled_points.append((r, c))

sampled_points = np.array(sampled_points)

# Now, relax those points using Lloyd's iteration

points = sampled_points
for i in range(100):
    f, ax = plt.subplots()

    old_points = points.copy()
    points = lloyd(points, boundary, ax=ax)

    ax.imshow(mask)
    ax.scatter(old_points[:, 1], old_points[:, 0], s=1)
    ax.scatter(points[:, 1], points[:, 0], s=1,
               color='red', marker='^', label='Adjusted points')
    ax.quiver(old_points[:, 1], old_points[:, 0],
              points[:, 1] - old_points[:, 1],
              points[:, 0] - old_points[:, 0],
              angles='xy', scale_units='xy', scale=1,
              units='dots', width=1, headwidth=10, headlength=10,
              color='gray')

    ax.set(xlim=[0, mask.shape[1]], ylim=[0, mask.shape[0]])
#    plt.savefig(f'/tmp/anim_{i:03d}.png', dpi=100)
#    plt.close()
    plt.show()

    print('.', end='', flush=True)

print()
