import numpy as np

class EnhancedRefinedAdaptiveMutationPSODE(RefinedAdaptiveMutationPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.line_change_prob = 0.2

    def __call__(self, func):
        population = self.initialize_population()
        for _ in range(self.budget):
            population = self.de_update(population, func)
        best_solution = min(population, key=lambda x: func(x['position']))
        return best_solution['position']