import numpy as np

class ImprovedDynamicABC:
    def __init__(self, budget, dim, mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]

            for i in range(self.budget):
                if i != best_idx:
                    mutation_vector = np.random.uniform(-1, 1, self.dim)
                    trial_solution = self.population[i] + mutation_vector * (best_solution - self.population[i]) * self.mutation_rate
                    if func(trial_solution) < fitness[i]:
                        self.population[i] = trial_solution
        return best_solution