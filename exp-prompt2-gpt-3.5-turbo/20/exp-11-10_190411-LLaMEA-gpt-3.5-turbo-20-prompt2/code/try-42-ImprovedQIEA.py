import numpy as np
from pyswarm import pso

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def objective(x):
            return func(np.array([x]))

        initial_guess = np.zeros(self.dim)
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        best_solution, _ = pso(objective, bounds[0], bounds[1], maxiter=self.budget, swarmsize=10)

        return best_solution