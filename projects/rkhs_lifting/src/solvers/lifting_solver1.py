
from projects.rkhs_lifting.src.solvers import Joint_Volume_Rots_Solver

class RKHS_Lifting_Solver1(Joint_Volume_Rots_Solver):
    def __init__(self):

        super().__init__()
        # Constuct problem
        # p = problem()

        # Construct options
        # o = options

        # https: // github.com / JuliaManifolds / Manopt.jl / blob / e0ec985b5baf177b5f7d0570899cb66d960f1199 / src / solvers / ChambollePock.jl  # L1-L60

    def stop_solver(self):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def stage1_step_solver(self):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def stage2_step_solver(self):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def volume_step(self):
        k=2

    def rots_density_step(self):
        k=2

    # TODO also forward etc in here