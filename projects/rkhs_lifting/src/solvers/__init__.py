

class Joint_Volume_Rots_Solver:
    def __init__(self):
        self.iter = 0

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

    def solve(self):
        while not self.stop_solver():
            self.iter += 1
            self.stage1_step_solver()

        self.stage2_step_solver()

        # TODO return result
