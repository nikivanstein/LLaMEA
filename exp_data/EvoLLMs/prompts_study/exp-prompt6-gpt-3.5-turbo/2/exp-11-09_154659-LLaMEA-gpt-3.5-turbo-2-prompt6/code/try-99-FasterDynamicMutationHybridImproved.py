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
                   'mutation_step': 1e-8} for _ in range(4)]  # Adjusted population size

        for i in range(2, self.budget - 3, 2):
            for wolf in wolves:
                local_search_position = nelder_mead_local_search(wolf, wolf['position'], wolf['mutation_step'])
                if func(local_search_position) < wolf['fitness']:
                    wolf['position'] = local_search_position
                    wolf['fitness'] = func(local_search_position)
                    adaptive_factor = 0.95 if func(local_search_position) < wolf['fitness'] else 1.05
                    wolf['mutation_step'] *= adaptive_factor  # Self-adaptive mutation based on individual performance
                    wolf['mutation_step'] *= 0.9  # Hybrid strategy combining individual and global search components

            # Dynamic population adjustment based on individual performance
            wolves = [wolf for wolf in wolves if func(wolf['position']) <= np.median([wolf['fitness'] for wolf in wolves])] + [{'position': np.random.uniform(-5.0, 5.0, self.dim),
                   'fitness': func(np.random.uniform(-5.0, 5.0, self.dim)),
                   'mutation_step': 1e-8}]

        best_wolf = min(wolves, key=lambda x: x['fitness'])
        return best_wolf['position']