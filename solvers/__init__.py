class Solver:
    def __init__(
            self,
            cost=None,
            max_it=200,
            tol=1 ** -4,
    ):
        if cost is None:
            raise RuntimeError("No cost function provided")
        else:
            self.cost = cost

        self.max_it = max_it
        self.tol = tol

        self.error_u = []
        self.relerror_u = []
        self.error_g = []
        self.relerror_g = []
        self.error_tot = []
        self.relerror_tot = []