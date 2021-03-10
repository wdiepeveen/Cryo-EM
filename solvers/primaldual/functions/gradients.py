import numpy as np
import odl

from aspire.volume import Volume

"""
The volumes are always normalized in range of (-1,1)^3
"""

def gradient_l2_mse_data_fidelity(src):
    """
    Computes gradient of
        1/2N \sum_i \| A g_i.u -f_i\|^2
    i.e.,
        1/N \sum_i A^* (A g_i.u - f_i)
    :param src:
    :return:
    """

    im = src.vol_forward(src.vols, src.rots) - src.data
    res = src.im_adjoint_forward(im)
    return Volume(res)

def gradient_l2_grad_norm(src):
    vol = src.vols.asnumpy()[0]
    discr = odl.uniform_discr([-1., -1., -1.], [1., 1., 1.], vol.shape)
    grad = odl.discr.diff_ops.Gradient(discr, pad_mode="order0")
    vol_grad = grad(vol)
    res = grad.adjoint(vol_grad).asarray()
    return Volume(res.astype(src.dtype))
    # return res.astype(src.dtype)