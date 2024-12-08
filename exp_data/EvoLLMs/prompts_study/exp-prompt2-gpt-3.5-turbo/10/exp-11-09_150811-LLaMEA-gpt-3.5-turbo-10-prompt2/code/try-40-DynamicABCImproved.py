import numpy as np

class DynamicABCImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        inertia_weight = 1.0
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]

            for i in range(self.budget):
                if i != best_idx:
                    trial_solution = self.population[i] + np.random.uniform(-1, 1, self.dim) * inertia_weight * (best_solution - self.population[i])
                    if func(trial_solution) < fitness[i]:
                        self.population[i] = trial_solution
            inertia_weight *= 0.99  # Decay the inertia weight for balancing exploration and exploitation
        return best_solution