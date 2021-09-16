import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse as sparse
from scipy.special import gammaln

from solvers.lifting.integration.icosahedron import IcosahedronIntegrator
from solvers.lifting.integration.rkhs.refined_mesh import RefinedMeshIntegrator

# We basically first want to see what this kernel looks like. For that we will at least need:
# - Integrator for the points
# - matrix with all the distances between the points
# - kernel function that takes the matrix as input and spits out W, W_t and W_d
# - plot kernel function for supervisors
# - plot sparse matrix with spy function + make plot

# Construct integrator
integrator = IcosahedronIntegrator()
quats = integrator.quaternions
print(quats.shape)

# Construct distance matrix between points
manifold = integrator.manifold
distances = manifold.dist(quats[None,:,:],quats[None,:,:])[0]
# distances = manifold.dist(quats,quats)
print(distances.shape)
print(distances)

# TODO kernel function
def cosine_kernel(omega, kappa=10):
    res = np.cos(omega/2)**(2*kappa)
    normalisation = np.sqrt(np.pi) * np.exp(gammaln(kappa+2) - gammaln(kappa + 1/2))
    return normalisation * res

kappa = 15
W = cosine_kernel(distances, kappa=kappa)
print(W)

# Plot kernel function for supervisors
omegas = np.linspace(0,np.pi,1000)
plt.plot(omegas, cosine_kernel(omegas, kappa=kappa))
plt.show()

# Plot sparse matrix with spy function + make plot
threshold = 1/integrator.n
Wt = W * (W >= threshold)

plt.spy(Wt)
plt.show()

# TODO construct RKHS_Integrator

integrator = RefinedMeshIntegrator(mesh_norm=0.5, base_integrator="icosahedron")

W = integrator.b2w
# Plot sparse matrix with spy function + make plot
threshold = 1/integrator.n
Wt = W * (W >= threshold)

plt.spy(Wt)
plt.show()

integrator = RefinedMeshIntegrator()

W = integrator.b2w
# Plot sparse matrix with spy function + make plot
threshold = 1/integrator.n
Wt = W * (W >= threshold)

plt.spy(Wt)
plt.show()