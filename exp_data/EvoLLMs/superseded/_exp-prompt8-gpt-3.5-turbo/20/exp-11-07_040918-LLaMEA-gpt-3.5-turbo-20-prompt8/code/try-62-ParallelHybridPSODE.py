import numpy as np
from scipy.optimize import differential_evolution
from joblib import Parallel, delayed

class ParallelHybridPSODE:
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

            def optimize_particle(agent):
                de_result = differential_evolution(objective, bounds=bounds_array, maxiter=5, popsize=popsize, init=agent)
                return de_result.x if de_result.fun < objective(best_pos) else best_pos

            for _ in range(self.budget // num_particles - 1):
                pop = [best_pos + np.random.normal(0, 1, self.dim) for _ in range(num_particles)]
                best_pos = np.array(Parallel(n_jobs=-1)(delayed(optimize_particle)(agent) for agent in pop))

            return best_pos

        return pso_de_optimizer()