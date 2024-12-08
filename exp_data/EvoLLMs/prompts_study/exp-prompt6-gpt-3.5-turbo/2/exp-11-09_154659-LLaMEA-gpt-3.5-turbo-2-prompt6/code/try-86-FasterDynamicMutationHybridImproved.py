import numpy as np
from scipy.optimize import minimize

class FasterDynamicMutationHybridImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def nelder_mead_local_search(wolf, initial_guess, mutation_step):
            res = minimize(func, initial_guess, method='Nelder-Mead', options={'xatol': mutation_step, 'disp': False})
            return res.x

        wolves = [{'position': np.random.uniform(-5.0, 5.0, self.dim),
                   'fitness': func(np.random.uniform(-5.0, 5.0, self.dim)),
                   'mutation_step': 1e-8} for _ in range(5)]

        for i in range(2, self.budget - 3, 2):
            for wolf in wolves:
                local_search_position = nelder_mead_local_search(wolf, wolf['position'], wolf['mutation_step'])
                if func(local_search_position) < wolf['fitness']:
                    wolf['position'] = local_search_position
                    wolf['fitness'] = func(local_search_position)
                    if np.random.rand() < 0.5:
                        wolf['mutation_step'] *= 0.95 if func(local_search_position) < wolf['fitness'] else 1.05
                    else:
                        wolf['mutation_step'] *= 0.9 * (1 - 0.1 * (func(local_search_position) < wolf['fitness']))
        
        best_wolf = min(wolves, key=lambda x: x['fitness'])
        return best_wolf['position']