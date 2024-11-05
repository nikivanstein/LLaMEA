class DEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(7 * (budget / 1000))

    def __call__(self, func):
        # DE optimization algorithm implementation with dynamic population size
        pass