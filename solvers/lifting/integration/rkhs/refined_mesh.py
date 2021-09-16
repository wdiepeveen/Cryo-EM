import numpy as np
import os

import itertools
import logging

import spherical

from scipy.sparse import csc_matrix
from scipy.spatial import ConvexHull

from solvers.lifting.integration.rkhs import RKHS_Integrator
from solvers.lifting.integration.sd1821 import SphDes1821Integrator
from solvers.lifting.integration.icosahedron import IcosahedronIntegrator

from solvers.lifting.integration.discretized_manifolds.so3 import SO3

logger = logging.getLogger(__name__)


class RefinedMeshIntegrator(RKHS_Integrator):

    def __init__(self,
                 mesh_norm=0.285298,  # TODO Is actually separation distance
                 # Maximal length of edges in the triangulation - is 0.285297 # 0.14621 on S3 | was 0.570595
                 kernel=10,
                 ell_max=5,
                 base_integrator="spherical-design",
                 dtype=np.float32,
                 ):

        angular_resolution = min(mesh_norm, 2 * np.pi / (ell_max + 2))
        print(angular_resolution)
        # second arg is minimum criterion for a quadrature scheme to exist

        # Construct base integrator
        if base_integrator == "spherical-design":
            quats = SphDes1821Integrator().quaternions
        elif base_integrator == "icosahedron":
            quats = IcosahedronIntegrator().quaternions
        else:
            raise NotImplementedError("This integrator is not available")

        manifold = SO3(quats=quats, h=angular_resolution)
        quaternions = manifold.verts

        # Initialize
        print("initialize super class")
        super().__init__(quaternions, kernel=15, dtype=dtype)

