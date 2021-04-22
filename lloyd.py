import numpy as np
from scipy import spatial
from shapely.geometry import Polygon


rng = np.random.default_rng()


def _points_contain_duplicates(points):
    """Check whether `points` contains duplicates.

    Parameters
    ----------
    points : (N, 2) ndarray of float
    """
    vals, count = np.unique(points, return_counts=True)
    return np.any(vals[count > 1])


def _jitter_points(points, scalar=.000000001):
    """Randomly jitter points until they are unique.

    If the points are not unique, the numberof regions in our
    tesselation will be less than the number of input points.
    """
    while _points_contain_duplicates(points):
        offset = rng.uniform(-scalar, scalar, size=(len(points), 2))
        points = points + offset
    return points


def lloyd(points, boundary, ax=None):
    """Perform Lloyd's iteration to space points within given boundary.

    Parameters
    ----------
    points : (N, 2) ndarray
        Row-column point coordinates.
    boundary : (N, 2) ndarray
        Row-column coordinates of boundary.
    ax : matplotlib axis, optional
        If provided, draw Voronoi diagram and point movement on given
        axis.

    Returns
    -------
    points : (N, 2) ndarray
        Points adjusted to centroids of Voronoi cells.

    References
    ----------
    https://en.wikipedia.org/wiki/Lloyd's_algorithm

    """
    # Close boundary polygon
    if not np.all(boundary[0] == boundary[-1]):
        boundary = np.vstack((boundary, boundary[0]))

    boundary = Polygon(boundary)

    points = _jitter_points(points)

    # Add exterior corners
    N_outside = 4
    points = np.vstack(
        (points,
         [
             [-10000, -10000],
             [-10000, 10000],
             [10000, 10000],
             [10000, -10000]
         ])
    )
    v = spatial.Voronoi(points)

    new_points = []

    for idx in v.point_region[:-N_outside]:
        # the region is a series of indices into self.voronoi.vertices
        # remove point at infinity, designated by index -1
        region = [i for i in v.regions[idx] if i != -1]

        # enclose the polygon
        region = region + [region[0]]

        # get the vertices for this region
        verts = v.vertices[region]

        if ax:
            ax.plot(verts[:, 1], verts[:, 0], color='green',
                    linewidth=1)

        # Clip region to outside polygon
        vpoly = Polygon(verts)
        ipoly = vpoly.intersection(boundary)

        if len(ipoly.centroid.coords) > 0:
            new_points.append(ipoly.centroid.coords[0])
        else:
            print('Warning: point on collapsed triangle?')
            if ax:
                ax.plot(verts[:, 1], verts[:, 0], color='magenta',
                        linewidth=1)
            # Point landed on a collapsed triangle?
            # Dither the point and try again
            new_points.append(points[idx] + 1e-6)

    new_points = np.array(new_points)

    return new_points
