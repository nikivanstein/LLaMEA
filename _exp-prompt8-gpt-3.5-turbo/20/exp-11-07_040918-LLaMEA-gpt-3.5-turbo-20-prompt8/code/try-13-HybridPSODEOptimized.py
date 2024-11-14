import numpy as np
from scipy.optimize import differential_evolution

class HybridPSODEOptimized:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def objective(x):
            return func(x)

        bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        popsize = 10
        num_particles = 5

        def pso_de_optimizer():
            bounds_array = np.array(bounds)
            pso_result = differential_evolution(objective, bounds=bounds_array, maxiter=5, popsize=popsize)
            best_pos = pso_result.x

            pop = np.zeros((self.budget // num_particles - 1, num_particles, self.dim))
            pop[0] = [best_pos + np.random.normal(0, 1, self.dim) for _ in range(num_particles)]
            for i in range(1, self.budget // num_particles - 1):
                for j in range(num_particles):
                    de_result = differential_evolution(objective, bounds=bounds_array, maxiter=5, popsize=popsize, init=pop[i-1, j])
                    if de_result.fun < objective(best_pos):
                        best_pos = de_result.x
                    pop[i, j] = de_result.x

            return best_pos

        return pso_de_optimizer()