import numpy as np
from scipy.optimize import differential_evolution

class HybridPSODEImproved:
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

            pop = [best_pos + np.random.normal(0, 1, (num_particles, self.dim)) for _ in range(self.budget // num_particles - 1)]
            for agents in pop:
                de_results = [differential_evolution(objective, bounds=bounds_array, maxiter=5, popsize=popsize, init=agent) for agent in agents]
                best_de_result = min(de_results, key=lambda x: x.fun)
                if best_de_result.fun < objective(best_pos):
                    best_pos = best_de_result.x

            return best_pos

        return pso_de_optimizer()