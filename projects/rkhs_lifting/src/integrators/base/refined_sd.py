import numpy as np
import os

from projects.rkhs_lifting.src.integrators.base import SO3_Integrator
from projects.rkhs_lifting.src.mesh.so3_midpoint_refinement import SO3_Midpoint_Refinement


class Refined_SD(SO3_Integrator):
    """Refined Spherical Design Integration"""

    def __init__(self,
                 base_integrator=None,
                 resolution=np.pi/10,  #0.285298,  # separation distance of the mesh
                 # Maximal length of edges in the triangulation - is 0.285297 # 0.14621 on S3 | was 0.570595
                 dtype=np.float32,):
        assert type(base_integrator) == SO3_Integrator

        refiner = SO3_Midpoint_Refinement(quats=base_integrator.quaternions, h=resolution)
        refined_integrator = SO3_Integrator(refiner.verts, dtype=dtype)

        super().__init__(refined_integrator.quaternions, dtype=dtype)
