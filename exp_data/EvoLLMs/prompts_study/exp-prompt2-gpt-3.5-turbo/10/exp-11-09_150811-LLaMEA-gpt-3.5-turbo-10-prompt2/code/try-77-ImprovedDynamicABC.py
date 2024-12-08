import numpy as np

class ImprovedDynamicABC:
    def __init__(self, budget, dim, mutation_factor=0.5):
        self.budget = budget
        self.dim = dim
        self.mutation_factor = mutation_factor
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]

            for i in range(self.budget):
                if i != best_idx:
                    mutation = np.random.uniform(0, 1, self.dim) * self.mutation_factor
                    trial_solution = self.population[i] + mutation * (best_solution - self.population[i])
                    if func(trial_solution) < fitness[i]:
                        self.population[i] = trial_solution
        return best_solution