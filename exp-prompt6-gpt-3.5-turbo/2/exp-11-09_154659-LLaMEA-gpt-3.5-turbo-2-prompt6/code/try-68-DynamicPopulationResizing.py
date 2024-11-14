import numpy as np
from scipy.optimize import minimize

class DynamicPopulationResizing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def nelder_mead_local_search(wolf, initial_guess):
            res = minimize(func, initial_guess, method='Nelder-Mead', options={'xatol': 1e-8, 'disp': False})
            return res.x

        wolves = [{'position': np.random.uniform(-5.0, 5.0, self.dim),
                   'fitness': func(np.random.uniform(-5.0, 5.0, self.dim))} for _ in range(5)]

        for i in range(2, self.budget - 3, 2):
            for wolf in wolves:
                local_search_position = nelder_mead_local_search(wolf, wolf['position'])
                if func(local_search_position) < wolf['fitness']:
                    wolf['position'] = local_search_position
                    wolf['fitness'] = func(local_search_position)
            
            if i % 10 == 0 and i < self.budget - 3:
                best_wolf = min(wolves, key=lambda x: x['fitness'])
                worst_wolf = max(wolves, key=lambda x: x['fitness'])
                new_wolf = {'position': best_wolf['position'], 'fitness': best_wolf['fitness']}
                wolves[wolves.index(worst_wolf)] = new_wolf

        best_wolf = min(wolves, key=lambda x: x['fitness'])
        return best_wolf['position']