import numpy as np
import odl

from aspire.image.image import Image
from aspire.utils import ainner
from aspire.utils.coor_trans import grid_3d
from aspire.volume import Volume

def compute_differential_forward_rot_ims(src, m):
    m = m.astype(src.dtype)
    vol = src.vols.asnumpy()[0]
    discr = odl.uniform_discr([-1., -1., -1.], [1., 1., 1.], vol.shape)
    grad = odl.discr.diff_ops.Gradient(discr, pad_mode="order0")
    u1, u2, u3 = grad(vol)

    u1 = u1.asarray()
    u2 = u2.asarray()
    u3 = u3.asarray()

    grid = grid_3d(src.L)

    x1 = grid["x"]
    x2 = grid["y"]
    x3 = grid["z"]

    # compute 6 u_j*x_k combinations
    vol12 = (u1 * x2).astype(src.dtype)
    vol13 = (u1 * x3).astype(src.dtype)
    vol23 = (u2 * x3).astype(src.dtype)
    vol21 = (u2 * x1).astype(src.dtype)
    vol31 = (u3 * x1).astype(src.dtype)
    vol32 = (u3 * x2).astype(src.dtype)

    # print("differentials.py Line 33: vol12 = {}".format(vol12))

    im12 = src.vol_forward(Volume(vol12), m)
    im13 = src.vol_forward(Volume(vol13), m)
    im23 = src.vol_forward(Volume(vol23), m)
    im21 = src.vol_forward(Volume(vol21), m)
    im31 = src.vol_forward(Volume(vol31), m)
    im32 = src.vol_forward(Volume(vol32), m)
    # print("differentials.py Line 41: im12 = {}".format(im12.asnumpy()[0]))

    return im12, im13, im23, im21, im31, im32



def differential_forward_rot(src, m, eta):
    if eta.shape != (src.n, 3, 3):
        raise RuntimeError("eta is not n-by-3-by-3")

    im12, im13, im23, im21, im31, im32 = compute_differential_forward_rot_ims(src, m)

    differential = (im12.data.T * eta[:, 0, 1]).T + (im13.data.T * eta[:, 0, 2]).T + (im23.data.T * eta[:, 1, 2]).T \
              + (im21.data.T * eta[:, 1, 0]).T + (im31.data.T * eta[:, 2, 0]).T + (im32.data.T * eta[:, 2, 1]).T

    return Image(-differential)


def adjoint_differential_forward_rot(src, m, xi):
    im12, im13, im23, im21, im31, im32 = compute_differential_forward_rot_ims(src, m)

    K12 = -1 / (src.L ** 2) * ainner(xi.data, im12.data, axes=(1, 2))
    K13 = -1 / (src.L ** 2) * ainner(xi.data, im13.data, axes=(1, 2))
    K23 = -1 / (src.L ** 2) * ainner(xi.data, im23.data, axes=(1, 2))

    K21 = -1 / (src.L ** 2) * ainner(xi.data, im21.data, axes=(1, 2))
    K31 = -1 / (src.L ** 2) * ainner(xi.data, im31.data, axes=(1, 2))
    K32 = -1 / (src.L ** 2) * ainner(xi.data, im32.data, axes=(1, 2))

    nu1 = 1 / np.sqrt(2) * (K12 - K21)
    nu2 = 1 / np.sqrt(2) * (K13 - K31)
    nu3 = 1 / np.sqrt(2) * (K23 - K32)

    nu = np.zeros((src.n, 3, 3))
    nu[:, 0, 1] = nu1
    nu[:, 0, 2] = nu2
    nu[:, 1, 2] = nu3
    nu[:, 1, 0] = -nu1
    nu[:, 2, 0] = -nu2
    nu[:, 2, 1] = -nu3

    adjoint = 1 / np.sqrt(2) * nu

    return Image(adjoint)
