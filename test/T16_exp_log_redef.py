import numpy as np
from solvers.lifting.integration.icosahedron import IcosahedronIntegrator

integrator = IcosahedronIntegrator()

p = integrator.quaternions
print(p.shape)
print(p)
X = -np.ones(p.shape)/3
pX = np.einsum("km,km->k", p,X)
X = X - pX[:,None] * p
print("X = {}".format(X))

q = integrator.manifold.exp(p,X)

# print(q.shape)
print(q)

XX = integrator.manifold.log(p,q)
# print("log = {}".format(XX))
print(np.max(np.abs(X-XX)))