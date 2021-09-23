

class Plan:

    def get_cost(self):
        raise NotImplementedError(
            "Subclasses should implement this and return a scalar"
        )