import numpy as np
import logging

from projects.rkhs_lifting.src.integrators.base import SO3_Integrator
from projects.rkhs_lifting.src.kernels import RKHS_Kernel
from projects.rkhs_lifting.src.integrators import RKHS_Density_Integrator
from projects.rkhs_lifting.src.mesh.so3_midpoint_refinement import SO3_Midpoint_Refinement

logger = logging.getLogger(__name__)


class Refined_Mesh(RKHS_Density_Integrator):

    def __init__(self,
                 base_integrator=None,
                 kernel=None,
                 resolution=np.pi/10,  #0.285298,  # separation distance of the mesh
                 # Maximal length of edges in the triangulation - is 0.285297 # 0.14621 on S3 | was 0.570595
                 dtype=np.float32,
                 ):

        assert type(base_integrator) == SO3_Integrator
        assert type(kernel) == RKHS_Kernel

        refiner = SO3_Midpoint_Refinement(quats=base_integrator.quaternions, h=resolution)
        refined_integrator = SO3_Integrator(refiner.verts, dtype=dtype)

        super().__init__(refined_integrator, kernel, dtype=dtype)

