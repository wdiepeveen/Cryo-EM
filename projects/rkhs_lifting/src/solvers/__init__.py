

class Joint_Volume_Rots_Solver:
    def __init__(self, plan):
        self.iter = 0
        self.cost = []
        self.plan = plan

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
        self.cost.append(self.plan.get_cost())  # TODO do this in an initialization
        while not self.stop_solver():
            self.iter += 1
            self.step_solver()

        self.finalize_solver()

        # TODO return result
