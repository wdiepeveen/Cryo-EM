import numpy as np
import os

from projects.rkhs_lifting.src.integrators.base import SO3_Integrator
from projects.rkhs_lifting.src.mesh.so3_midpoint_refinement import SO3_Midpoint_Refinement


class Refined_SD(SO3_Integrator):
    """Refined Spherical Design Integration"""

    def __init__(self,
                 base_integrator=None,
                 resolution=np.pi/5,  # > 0.570596, which is the separation distance of the SD1821 design
                 dtype=np.float32,
                 ):
        assert issubclass(type(base_integrator), SO3_Integrator)

        refiner = SO3_Midpoint_Refinement(quats=base_integrator.quaternions, h=resolution)
        # refined_integrator = SO3_Integrator(refiner.verts, dtype=dtype)

        super().__init__(refiner.verts, dtype=dtype)
