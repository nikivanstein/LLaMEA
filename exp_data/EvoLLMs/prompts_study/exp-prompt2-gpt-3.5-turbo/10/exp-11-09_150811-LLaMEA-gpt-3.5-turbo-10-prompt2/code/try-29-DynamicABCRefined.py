import numpy as np

class DynamicABCRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]

            for i in range(self.budget):
                if i != best_idx:
                    if np.random.rand() < 0.1:
                        trial_solution = np.random.uniform(-5.0, 5.0, self.dim)
                    else:
                        trial_solution = self.population[i] + np.random.uniform(-1, 1, self.dim) * (best_solution - self.population[i])
                    if func(trial_solution) < fitness[i]:
                        self.population[i] = trial_solution
        return best_solution