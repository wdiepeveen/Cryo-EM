import numpy as np
import odl

from pymanopt.manifolds import Rotations

from scipy.linalg import norm

"""
The volumes are always normalized in range of (-1,1)
"""

def l2_data_fidelity(src):
    # Compute res
    res = src.vol_forward(src.vols, 0, np.inf) - src.data
    # Compute error
    cost = 1 / (2 * src.n) * (2 / src.L) ** 2 * norm(res.data) ** 2
    return cost


def l2_grad_norm(src):
    vol = src.vols.asnumpy()[0]
    # discr = odl.uniform_discr([0, 0, 0], [src.L, src.L, src.L], vol.shape)
    discr = odl.uniform_discr([-1., -1., -1.], [1., 1., 1.], vol.shape)
    grad = odl.discr.diff_ops.Gradient(discr, pad_mode="order0")
    vol_grad = grad(vol)
    # cost = 1 / (2 * src.L) * norm(vol_grad) ** 2
    cost = 1 / 2 * (2 / src.L)**3 * norm(vol_grad) ** 2
    return cost


def so3_distance(src, p=2):
    manifold = Rotations(3, src.n)
    cost = 1 / (p * src.n) * manifold.dist(src.rots, src.init_rots) ** p
    return cost
