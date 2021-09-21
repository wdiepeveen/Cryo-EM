import numpy as np
import logging


from solvers.lifting.integration.rkhs import RKHS_Integrator
from solvers.lifting.integration.sd1821 import SphDes1821Integrator
from solvers.lifting.integration.icosahedron import IcosahedronIntegrator

from solvers.lifting.integration.discretized_manifolds.so3 import SO3

logger = logging.getLogger(__name__)


class RefinedMeshIntegrator(RKHS_Integrator):

    def __init__(self,
                 separation=0.285298,  # separation distance of the mesh
                 # Maximal length of edges in the triangulation - is 0.285297 # 0.14621 on S3 | was 0.570595
                 base_integrator=1821,
                 dtype=np.float32,
                 ):

        # Construct base integrator
        if base_integrator == 1821:
            quats = SphDes1821Integrator().quaternions
        elif base_integrator == 60:
            quats = IcosahedronIntegrator().quaternions
        elif base_integrator == 300:
            quats = IcosahedronIntegrator().quaternions
        else:
            raise NotImplementedError("This integrator is not available: base_integrator does not have value in [60,300,1821]")

        manifold = SO3()

        # Initialize
        print("initialize super class")
        super().__init__(quats, kernel=15, dtype=dtype)

