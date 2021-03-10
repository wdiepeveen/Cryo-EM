import numpy as np

from aspire.utils import anorm, ainner

from pymanopt.manifolds import Rotations

def conj_l2_mse_data_fidelity(src, xi, b=1):
    """
    conjugate of F(x) = 1/(2N\sigma^2) \sum_i \|x_i - f_i\|^2 and for G(x) = bF(x)
    :param src:
    :return:
    """
    conj = src.noise_var * src.n / (2 * b) * np.sum(anorm(xi.data, axes=(1, 2))) \
           + np.sum(ainner(xi.data, src.data, axes=(1, 2)))
    conj = (2 / src.L) ** 2 * conj  # adjust for grid spacing/integration

    return conj

def conj_so3_distance(src, xi, b=1):
    conj = src.n / (2 * b) * np.sum(anorm(xi.data, axes=(1, 2)))

    conj = (2 / src.L) ** 2 * conj  # adjust for grid spacing/integration

    return conj