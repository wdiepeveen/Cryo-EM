

class Joint_Volume_Rots_Solver:
    def __init__(self):
        self.iter = 0

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
        while not self.stop_solver():
            self.iter += 1
            self.step_solver()

        self.finalize_solver()

        # TODO return result
