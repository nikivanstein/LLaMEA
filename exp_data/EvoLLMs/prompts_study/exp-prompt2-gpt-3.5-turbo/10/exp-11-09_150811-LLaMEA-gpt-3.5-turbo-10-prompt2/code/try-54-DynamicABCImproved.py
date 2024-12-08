import numpy as np

class DynamicABCImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.scale_factor = 0.5
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]

            for i in range(self.budget):
                if i != best_idx:
                    scale = np.exp(-_ / self.budget)  # Dynamic mutation strategy
                    trial_solution = self.population[i] + np.random.uniform(-1, 1, self.dim) * (best_solution - self.population[i]) * scale
                    if func(trial_solution) < fitness[i]:
                        self.population[i] = trial_solution
        return best_solution