import numpy as np
from scipy.optimize import differential_evolution

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def objective(x):
            return func(x)

        bounds = np.tile([(-5.0, 5.0)], (self.dim, 1))
        popsize = 10
        num_particles = 5

        def pso_de_optimizer():
            pso_result = differential_evolution(objective, bounds=bounds, maxiter=5, popsize=popsize)
            best_pos = pso_result.x

            for _ in range(self.budget // num_particles - 1):
                pop = np.tile(best_pos, (num_particles, 1)) + np.random.normal(0, 1, (num_particles, self.dim))
                for agent in pop:
                    de_result = differential_evolution(objective, bounds=bounds, maxiter=5, popsize=popsize, init=agent)
                    if de_result.fun < objective(best_pos):
                        best_pos = de_result.x

            return best_pos

        return pso_de_optimizer()