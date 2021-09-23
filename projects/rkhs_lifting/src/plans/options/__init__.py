

class Options:

    def __init__(self, stop=None):
        self.stop = stop

    def get_solver_result(self):
        raise NotImplementedError(
            "Subclasses should implement this and return a Volume object"
        )

    # TODO think about debug options and options in general