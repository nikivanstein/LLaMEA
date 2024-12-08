import numpy as np
from scipy.optimize import minimize

class MultiStartFasterDynamicMutationHybridImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def nelder_mead_local_search(wolf, initial_guess, mutation_step):
            res = minimize(func, initial_guess, method='Nelder-Mead', options={'xatol': mutation_step, 'disp': False})
            return res.x

        populations = [[[np.random.uniform(-5.0, 5.0, self.dim),
                        func(np.random.uniform(-5.0, 5.0, self.dim)),
                        1e-8] for _ in range(4)]]  # Initialize multiple independent populations

        for i in range(2, self.budget - 3, 2):
            for population in populations:
                for wolf in population:
                    local_search_position = nelder_mead_local_search(wolf, wolf[0], wolf[2])
                    if func(local_search_position) < wolf[1]:
                        wolf[0] = local_search_position
                        wolf[1] = func(local_search_position)
                        adaptive_factor = 0.95 if func(local_search_position) < wolf[1] else 1.05
                        wolf[2] *= adaptive_factor
                        wolf[2] *= 0.9

                population[:] = [wolf for wolf in population if func(wolf[0]) <= np.median([w[1] for w in population])] + [[np.random.uniform(-5.0, 5.0, self.dim),
                                func(np.random.uniform(-5.0, 5.0, self.dim)),
                                1e-8]]

        best_wolf = min([min(population, key=lambda x: x[1]) for population in populations], key=lambda x: x[1])
        return best_wolf[0]