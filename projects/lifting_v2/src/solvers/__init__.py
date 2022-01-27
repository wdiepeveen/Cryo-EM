

class Joint_Volume_Rots_Solver:
    def __init__(self, plan):
        self.iter = 0
        self.cost = []
        self.plan = plan

        self.vol_iterates = []
        self.rots_coeffs_iterates = []
        self.sigmas_iterates = []
        self.tau_iterates = []

    def initialize_solver(self):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def stop_solver(self):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def step_solver(self):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def finalize_solver(self):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def solve(self):
        print("Initializing solver")
        self.initialize_solver()
        # self.cost.append(self.plan.get_cost())  # TODO do this in an initialization
        print("Start solver")
        while not self.stop_solver():
            self.iter += 1
            print("========================= Iteration {} =========================".format(self.iter))
            self.step_solver()
            # print("=============== Iteration 1: Cost = {} ===============".format(self.cost[-1]))

        self.finalize_solver()

        # TODO return result
