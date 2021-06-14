import numpy as np
from scipy.spatial import Delaunay

from solvers.lifting.integration.refined_mesh import RefinedMeshIntegrator


points = np.array([[0, 0, 0], [0, 1.1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1.1, 1], [1, 0, 1], [1, 1, 1]])
tri = Delaunay(points)
print(tri.simplices)

# TODO we need convex hull triangulation
# We also need a way to check whether indeed all the points have been included
# We only need the convex hull for initialization, after that we can throw it away
# and only work with the simplices on the right side of the sphere. In that case we can continue on Thomas' script!

# we cannot just give rotations as input. We should start with a full S3 design --> use the 1821 (3642)

integrator = RefinedMeshIntegrator(ell_max=15, mesh_norm=np.pi/20, mesh_norm2=np.pi/10)  #, base_integrator="icosahedron")