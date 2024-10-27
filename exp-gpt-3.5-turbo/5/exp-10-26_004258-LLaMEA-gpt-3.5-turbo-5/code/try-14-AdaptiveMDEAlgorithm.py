import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveMDEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def objective(x):
            return func(x)

        def optimize_adaptive_mde(x0):
            bounds = [(-5, 5)] * self.dim
            mutation_factors = np.random.uniform(0.5, 1.5, self.dim)  # Introducing adaptive mutation factors

            def adaptive_mutation_mutation(x, xr, r, f):
                return x + f * mutation_factors * (xr - x)

            result = differential_evolution(objective, bounds, maxiter=self.budget, seed=42, popsize=10, tol=0.01, mutation=adaptive_mutation_mutation)
            return result.x, result.fun

        x0 = np.random.uniform(-5, 5, self.dim)

        best_solution = optimize_adaptive_mde(x0)
        
        return best_solution