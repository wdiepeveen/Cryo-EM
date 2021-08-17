import numpy as np
import os

import itertools
import logging

import spherical

from scipy.sparse import csc_matrix
from scipy.spatial import ConvexHull

from solvers.lifting.integration import Integrator
from solvers.lifting.integration.sd1821 import SphDes1821Integrator
from solvers.lifting.integration.icosahedron import IcosahedronIntegrator

from solvers.lifting.integration.discretized_manifolds.so3 import SO3

logger = logging.getLogger(__name__)


class RefinedMeshIntegrator(Integrator):

    def __init__(self,
                 ell_max=3,  # Sets the angular resolution
                 mesh_norm=0.285298,
                 # Maximal length of edges in the triangulation - is 0.285297 # 0.14621 on S3 | was 0.570595
                 base_integrator="spherical-design",
                 dtype=np.float32,
                 ):

        angular_resolution = min(mesh_norm, 2 * np.pi / (ell_max + 2))
        # second arg is minimum criterion for a quadrature scheme to exist

        # Construct base integrator
        if base_integrator == "spherical-design":
            quats = SphDes1821Integrator().quaternions
        elif base_integrator == "icosahedron":
            quats = IcosahedronIntegrator().quaternions
        else:
            raise NotImplementedError("This integrator is not available")

        manifold = SO3(quats=quats, h=angular_resolution)

        verts = manifold.verts

        n = verts.shape[0]

        # Initialize
        super().__init__(dtype=dtype, n=n, ell_max=ell_max, t=np.inf)

        self.quaternions = verts
        self.manifold = manifold
        self.initialize_b2w()
