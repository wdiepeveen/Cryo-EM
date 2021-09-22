class DensityUpdate:
    def __init__(self, problem):
        self.problem = problem
        self.dtype = problem.dtype

        self.ell = problem.integrator.ell
        self.ng = problem.integrator.n

        self.L = problem.L
        self.n = problem.n

    def update(self):
        raise NotImplementedError(
            "Subclasses should implement this and return a Volume object"
        )