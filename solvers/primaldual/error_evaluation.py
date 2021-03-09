from pymanopt.manifolds import Rotations

def get_rots_manifold_mse(rots_reg, rots_ref):
    K = rots_reg.shape[0]
    manifold = Rotations(3, K)

    dist = 1 / K * manifold.dist(rots_reg, rots_ref)**2
    return dist