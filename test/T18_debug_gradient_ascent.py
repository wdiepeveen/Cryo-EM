import numpy as np
import spherical
import matplotlib.pyplot as plt

from solvers.lifting.integration.hexacosichoron import HexacosichoronIntegrator
from solvers.lifting.integration.sd1821 import SphDes1821Integrator
from solvers.lifting.integration.uniform import UniformIntegrator
from solvers.lifting.integration.refined_mesh import RefinedMeshIntegrator

from solvers.lifting.functions.rot_converters import quat2angle

from T17_density_gradient_ascent import gradient_ascent


# integrator = SphDes1821Integrator(ell_max=10)
# integrator = UniformIntegrator(ell_max=10, n=20000)
integrator = RefinedMeshIntegrator(ell_max=10, mesh_norm=np.pi/20)

wigner = spherical.Wigner(integrator.ell_max)
l = 5
m = 4
n = 1
index = wigner.Dindex(l, m, n)
coeffs = np.eye(integrator.ell)[index][None]
print(coeffs.shape)

density = integrator.coeffs2weights(coeffs)
# Plot density
angles = integrator.angles

x = angles[:, 0]
y = angles[:, 1]
z = angles[:, 2]
c = density[0, :] * len(x)  # only first density for visualization
print("Number of angles = {}".format(len(x)))

fig = plt.figure()
ax = fig.gca(projection='3d')

img = ax.scatter(x, y, z, c=c, cmap=plt.cool())
ax.set_xlabel("$\phi$")
ax.set_ylabel("$\\theta$")
ax.set_zlabel("$\psi$")  #, rotation=0)
plt.colorbar(img)

plt.show()

start = integrator.quaternions[np.argmax(density, axis=-1),:]
print("start = {}".format(start))
print("cost at start = {}".format(np.max(density, axis=-1)* len(x)))
distortion = np.array([1e-4, 1e-4, 1e-4, 1e-4])[None]
proj_distortion = np.einsum("ik,ik->i", start, distortion)
v = distortion - proj_distortion[:, None] * start

# start = integrator.manifold.exp(start, v)
print("distorted start = {}".format(start))

sols, costs = gradient_ascent(integrator,start,coeffs,1e-6,max_iter=200)
print("sols = {}".format(sols))

print("distance between start and sol = {}".format(integrator.manifold.dist(start,sols)))

plt.plot(range(len(costs)), costs)
plt.show()

angles = quat2angle(sols)

print("phi = {}, theta = {}, psi = {}".format(angles[0,0], angles[0,1], angles[0,2]))