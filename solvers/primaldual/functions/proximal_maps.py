import numpy as np

from aspire.image.image import Image
from pymanopt.manifolds import Rotations


def prox_l2_data_fidelity(src, a, x):
    prox_res = 1 / (1 + a) * (x + a * src.data)
    return prox_res


def prox_dual_l2_mse_data_fidelity(src, a, xi):
    dual_prox_res = 1 / (1 + a * src.n) * (xi.asnumpy() - a * src.data.asnumpy())
    return Image(dual_prox_res)


def prox_so3_distance(src, a, x, p=2):
    manifold = Rotations(3, src.n)
    d = manifold.dist(x, src.init_rots)
    if p == 2:
        t = a / (1 + a)
    elif p == 1:
        if a < d:
            t = a / d
        else:
            t = 1.0
    else:
        NotImplementedError(
            f"Proximal Map of distance(M,f,x) not implemented for p={p} (requires p=1 or 2)"
        )

    prox_res = manifold.exp(x, t * manifold.log(x, src.init_rots))

    return prox_res


def prox_constraint_so3_distance(src, a, x, p=2, radius=np.pi/2):

    assert radius <= np.pi/2

    manifold = Rotations(3, src.n)
    d = manifold.dist(x, src.init_rots)
    if p == 2:
        t = a / (1 + a)
    elif p == 1:
        if a < d:
            t = a / d
        else:
            t = 1.0
    else:
        NotImplementedError(
            f"Proximal Map of distance(M,f,x) not implemented for p={p} (requires p=1 or 2)"
        )

    # Project back onto constraint set with radius
    base_manifold = Rotations(3)
    tlog = np.zeros((src.n, 3, 3))
    for i in range(src.n):
        d_i = base_manifold.dist(x[i], src.init_rots[i])
        if (1-t)* d_i > radius:
            t_i = 1 - radius / d_i
            tlog[i] = t_i * base_manifold.log(x[i], src.init_rots[i])
        else:
            tlog[i] = t * base_manifold.log(x[i], src.init_rots[i])

    prox_res = manifold.exp(x, tlog)

    return prox_res
