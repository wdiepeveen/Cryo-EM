import numpy as np
import os

from projects.lifting_v2.src.integrators.base.sd1821mrx import SD1821MRx
from projects.lifting_v2.src.solvers.refinement_solver2 import Refinement_Solver2


def projection_simplex(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    # Function from: https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
    # Author: Mathieu Blondel
    """
    if axis == 1:
        # print("len(V) = {}".format(len(V)))
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        # print("U = {}".format(U[0]))
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        # print("cssv = {}".format(cssv[1]))
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()


# TODO constuct large matrices and see when it breaks down
n = int(1e6)
N = 100
matrix = np.random.rand(n,N)
rots_coefficients = projection_simplex(matrix, axis=0)
print("done")


