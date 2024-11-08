import numpy as np
from scipy.optimize import differential_evolution

class ImprovedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        self.popsize = 10
        self.num_particles = 5

    def __call__(self, func):
        def objective(x):
            return func(x)

        bounds_array = np.array(self.bounds)

        def pso_de_optimizer():
            pso_result = differential_evolution(objective, bounds=bounds_array, maxiter=5, popsize=self.popsize)
            best_pos = pso_result.x

            for _ in range(self.budget // self.num_particles - 1):
                pop = [best_pos + np.random.normal(0, 1, self.dim) for _ in range(self.num_particles)]
                for agent in pop:
                    de_result = differential_evolution(objective, bounds=bounds_array, maxiter=5, popsize=self.popsize, init=agent)
                    if de_result.fun < objective(best_pos):
                        best_pos = de_result.x

            return best_pos

        return pso_de_optimizer()