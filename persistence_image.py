
import numpy as np
import matplotlib.pyplot as plt
import math
import gudhi as gd


# Precompute static gaussian cdf lookup table
_ERF_CLIP = 8.0
_ERF_SAMPLES = 4097
_ERF_X = np.linspace(-_ERF_CLIP, _ERF_CLIP, _ERF_SAMPLES)
_ERF_Y = np.array([math.erf(x) for x in _ERF_X], dtype=float)


def _erf_interp(x):

    x = np.clip(x, -_ERF_CLIP, _ERF_CLIP)
    return np.interp(x, _ERF_X, _ERF_Y)


def weight_func(b, l, w):
            if w <= 0:
                return 1

            if l<= w:
                return l / w
            else:
                return 1
#persistence diagram of a point cloud/sublevel set filtration
class PD:
    def __init__(self, points, max_dimension=2, max_edge_length=np.inf,sparse=None, type='point_cloud'):

        self.points = points
        self.type = type
        if self.type == 'point_cloud':
            self.rips_complex = gd.RipsComplex(points=self.points, max_edge_length=max_edge_length, sparse=sparse)
            self.simplex_tree = self.rips_complex.create_simplex_tree(max_dimension=2)
            self.persistence_diagram = self.simplex_tree.persistence()
        elif self.type == 'sublevel_set':

            if isinstance(self.points, np.ndarray):
                u = self.points
            elif isinstance(self.points, (tuple, list)):
                if len(self.points) >= 3:
                    u = self.points[2]
                else:
                    raise ValueError(
                        "For type='sublevel_set', points must be a 2D array or a tuple/list with at least 3 items."
                    )
            else:
                raise ValueError(
                    "For type='sublevel_set', points must be a 2D array or tuple/list input."
                )

            u = np.asarray(u)
            if u.ndim != 2:
                raise ValueError(
                    f"For type='sublevel_set', u must be 2D, got shape {u.shape}."
                )

            self.cubical_complex = gd.CubicalComplex(top_dimensional_cells=u)
            self.persistence_diagram = self.cubical_complex.persistence()




    def PI(self, w, res=(20, 20), distribution='gaussian', sigma=0.001, box=None):
        """Compute persistence image from persistence diagram.

        Args:
            w: Weight parameter for persistence points
            res: Resolution (res_x, res_y) of the output persistence image grid. Default: (20, 20)
            distribution: Type of distribution.
            sigma: std for gaussian.
            box: Bounding box for the persistence diagram. Can be:
                - None (default): Adaptive bounding box based on data (95th percentile + 3*sigma)
                - 4-tuple (x_min, x_max, y_min, y_max): Same box for both H0 and H1
                - Pair of 4-tuples ((x_min0, x_max0, y_min0, y_max0), (x_min1, x_max1, y_min1, y_max1)):
                  Separate boxes for H0 and H1 respectively

        Returns:
            Tuple of (vector, pi_0, pi_1, bl_dim0, bl_dim1, extent_0, extent_1)
        """
        pd = self.persistence_diagram
        bl_points_dim0 = []
        bl_points_dim1 = []

        for item in pd:
            if len(item) != 2:
                continue

            dim, pair = item
            if not (isinstance(pair, (tuple, list, np.ndarray)) and len(pair) == 2):
                continue

            birth, death = pair
            if death == float('inf'):
                continue

            lifetime = death - birth
            if dim == 0:
                bl_points_dim0.append([birth, lifetime])
            elif dim == 1:
                bl_points_dim1.append([birth, lifetime])

        bl_dim0 = np.array(bl_points_dim0, dtype=float) if bl_points_dim0 else np.empty((0, 2), dtype=float)
        bl_dim1 = np.array(bl_points_dim1, dtype=float) if bl_points_dim1 else np.empty((0, 2), dtype=float)


        weights_dim0 = (
            np.array([weight_func(b, l, w) for b, l in bl_dim0], dtype=float)
            if bl_dim0.shape[0] > 0
            else np.empty(0, dtype=float)
        )
        weights_dim1 = (
            np.array([weight_func(b, l, w) for b, l in bl_dim1], dtype=float)
            if bl_dim1.shape[0] > 0
            else np.empty(0, dtype=float)
        )

        inv_sigma_sqrt2 = 1.0 / (sigma * np.sqrt(2.0))

        def cdf(z, bl_dim, weights):
            if bl_dim.shape[0] == 0:
                return 0.0

            x_term = 0.5 * (1.0 + _erf_interp((z[0] - bl_dim[:, 0]) * inv_sigma_sqrt2))
            y_term = 0.5 * (1.0 + _erf_interp((z[1] - bl_dim[:, 1]) * inv_sigma_sqrt2))
            return float(np.sum(weights * x_term * y_term))

        # Build separate birth-lifetime boxes for H0 and H1

        box_0 = None
        box_1 = None
        if box is not None:
            if len(box) == 2 and len(box[0]) == 4 and len(box[1]) == 4:
                box_0, box_1 = box
            elif len(box) == 4:
                box_0 = box
                box_1 = box
            else:
                raise ValueError("box must be either a 4-tuple or a pair of two 4-tuples.")
        #data adaptive box-size
        def make_extent(bl_dim, custom_box):
            if custom_box is not None:
                x_min, x_max, y_min, y_max = custom_box
            elif bl_dim.shape[0] > 0:
                x_min = 0.0
                y_min = 0.0
                x_max = np.percentile(bl_dim[:, 0], 95) + 3 * sigma
                y_max = np.percentile(bl_dim[:, 1], 95) + 3 * sigma
            else:
                x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0

            x_max = max(float(x_max), float(x_min) + 1e-9)
            y_max = max(float(y_max), float(y_min) + 1e-9)
            return (float(x_min), float(x_max), float(y_min), float(y_max))

        extent_0 = make_extent(bl_dim0, box_0)
        extent_1 = make_extent(bl_dim1, box_1)
        x_bins_0 = np.linspace(extent_0[0], extent_0[1], res[0] + 1)
        y_bins_0 = np.linspace(extent_0[2], extent_0[3], res[1] + 1)
        x_bins_1 = np.linspace(extent_1[0], extent_1[1], res[0] + 1)
        y_bins_1 = np.linspace(extent_1[2], extent_1[3], res[1] + 1)

        # persistence pixel computation
        pi_0 = np.zeros(res)
        pi_1 = np.zeros(res)

        if bl_dim0.shape[0] > 0:
            for i in range(res[0]):
                for j in range(res[1]):
                    z1_0 = (x_bins_0[i], y_bins_0[j])
                    z2_0 = (x_bins_0[i + 1], y_bins_0[j + 1])
                    pi_0[i, j] = (
                        cdf(z2_0, bl_dim0, weights_dim0)
                        - cdf((z1_0[0], z2_0[1]), bl_dim0, weights_dim0)
                        - cdf((z2_0[0], z1_0[1]), bl_dim0, weights_dim0)
                        + cdf(z1_0, bl_dim0, weights_dim0)
                    )

        if bl_dim1.shape[0] > 0:
            for i in range(res[0]):
                for j in range(res[1]):
                    z1_1 = (x_bins_1[i], y_bins_1[j])
                    z2_1 = (x_bins_1[i + 1], y_bins_1[j + 1])
                    pi_1[i, j] = (
                        cdf(z2_1, bl_dim1, weights_dim1)
                        - cdf((z1_1[0], z2_1[1]), bl_dim1, weights_dim1)
                        - cdf((z2_1[0], z1_1[1]), bl_dim1, weights_dim1)
                        + cdf(z1_1, bl_dim1, weights_dim1)
                    )

        return np.concatenate((pi_0.flatten(), pi_1.flatten())), pi_0, pi_1, bl_dim0, bl_dim1, extent_0, extent_1


def plot_PI_heatmap(pi, res=(20, 20)):

    if len(pi) == 7:
        vec, pi_0, pi_1, bl_dim0, bl_dim1, extent_0, extent_1 = pi
    elif len(pi) == 6:
        vec, pi_0, pi_1, bl_dim0, bl_dim1, fixed_extent = pi
        extent_0 = fixed_extent
        extent_1 = fixed_extent
    else:
        vec, pi_0, pi_1, bl_dim0, bl_dim1 = pi
        extent_0 = (0.0, 1.0, 0.0, 1.0)
        extent_1 = (0.0, 1.0, 0.0, 1.0)

    # plot against separate birth-lifetime boxes for H0 and H1
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axs[0].imshow(pi_0.T, origin='lower', aspect='auto', extent=extent_0)
    axs[0].set_title('Persistence Image (H0)')
    axs[0].set_xlabel('Birth')
    axs[0].set_ylabel('Lifetime')
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(pi_1.T, origin='lower', aspect='auto', extent=extent_1)
    axs[1].set_title('Persistence Image (H1)')
    axs[1].set_xlabel('Birth')
    axs[1].set_ylabel('Lifetime')
    fig.colorbar(im1, ax=axs[1])
    plt.tight_layout()
    return fig, axs
