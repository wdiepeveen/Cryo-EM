import numpy as np
import os

from projects.lifting_v2.src.integrators.base import SO3_Integrator
from projects.lifting_v2.src.mesh.midpoint_refinement import Midpoint_Refinement


class Refined_SD(SO3_Integrator):
    """Refined Spherical Design Integration"""

    def __init__(self,
                 base_integrator=None,
                 resolution=np.pi/5,  # > 0.570596, which is the separation distance of the SD1821 design
                 dtype=np.float32,
                 ):
        assert issubclass(type(base_integrator), SO3_Integrator)

        refiner = Midpoint_Refinement(quats=base_integrator.quaternions, h=resolution)
        # refined_integrator = SO3_Integrator(refiner.verts, dtype=dtype)

        super().__init__(refiner.verts, dtype=dtype)