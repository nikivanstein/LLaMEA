import numpy as np
from scipy.optimize import minimize

class FasterDynamicMutationHybridImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def differential_evolution_mutation(wolf, population, differential_weight):
            candidate = population[np.random.choice(len(population))]
            mutated_wolf = wolf['position'] + differential_weight * (candidate['position'] - wolf['position'])
            return mutated_wolf

        wolves = [{'position': np.random.uniform(-5.0, 5.0, self.dim),
                   'fitness': func(np.random.uniform(-5.0, 5.0, self.dim)),
                   'mutation_step': 1e-8} for _ in range(4)]

        for i in range(2, self.budget - 3, 2):
            for wolf in wolves:
                mutated_position = differential_evolution_mutation(wolf, wolves, 0.5)
                local_search_position = nelder_mead_local_search(wolf, mutated_position, wolf['mutation_step'])

                if func(local_search_position) < wolf['fitness']:
                    wolf['position'] = local_search_position
                    wolf['fitness'] = func(local_search_position)
                    adaptive_factor = 0.95 if func(local_search_position) < wolf['fitness'] else 1.05
                    wolf['mutation_step'] *= adaptive_factor
                    wolf['mutation_step'] *= 0.9

            wolves = [wolf for wolf in wolves if func(wolf['position']) <= np.median([wolf['fitness'] for wolf in wolves])] + [{'position': np.random.uniform(-5.0, 5.0, self.dim),
                   'fitness': func(np.random.uniform(-5.0, 5.0, self.dim)),
                   'mutation_step': 1e-8}]

        best_wolf = min(wolves, key=lambda x: x['fitness'])
        return best_wolf['position']