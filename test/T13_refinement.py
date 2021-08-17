import numpy as np
import matplotlib.pyplot as plt
from solvers.lifting.integration.refined_mesh import RefinedMeshIntegrator
from solvers.lifting.integration.icosahedron import IcosahedronIntegrator

# Construct integrator
integrator = RefinedMeshIntegrator(ell_max=0, mesh_norm=np.pi/5, base_integrator="icosahedron")
# integrator = IcosahedronIntegrator()

print(integrator.quaternions)
# Plot refined points
angles = integrator.angles

x = angles[:, 0]
y = angles[:, 1]
z = angles[:, 2]
print("Number of angles = {}".format(len(x)))

fig = plt.figure()
ax = fig.gca(projection='3d')

img = ax.scatter(x, y, z)
ax.set_xlabel("$\phi$")
ax.set_ylabel("$\\theta$")

# ax.zaxis.set_rotate_label(False)
ax.set_zlabel("$\psi$")  #, rotation=0)
plt.colorbar(img)

plt.show()