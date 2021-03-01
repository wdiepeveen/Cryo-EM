from aspire.image.image import Image
from pymanopt.manifolds import Rotations


def prox_l2_data_fidelity(src, a, x):
    prox_res = 1 / (1 + a) * (x + a * src.data)
    return prox_res


def prox_dual_l2_data_fidelity(src, a, xi):
    dual_prox_res = 1 / (1 + a * src.n) * (xi.data - a * src.data.data)
    return Image(dual_prox_res)


def prox_so3_distance(src, a, x, p=2):
    manifold = Rotations(3, src.n)
    a = a/src.n
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
