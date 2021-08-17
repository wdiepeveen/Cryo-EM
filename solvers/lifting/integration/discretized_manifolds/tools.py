import numpy as np

def normalize(u, p=2, thresh=0.0):
    """ Normalizes u along the last axis with norm p.
    If  |u| <= thresh, 0 is returned (this mimicks the sign function).
    """
    ndim = u.shape[-1]
    multi = u.shape if u.ndim > 1 else None
    u = u.reshape(1,ndim) if multi is None else u.reshape(-1,ndim)
    ns = np.linalg.norm(u, ord=p, axis=1)
    fact = np.zeros_like(ns)
    fact[ns > thresh] = 1.0/ns[ns > thresh]
    out = fact[:,None]*u
    return out[0] if multi is None else out.reshape(multi)