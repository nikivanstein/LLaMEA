import numpy as np

class DynamicABCRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def mutation_operator(self, x, best_solution):
        return x + np.random.uniform(-1, 1, self.dim) * (best_solution - x)
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(x) for x in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]

            for i in range(self.budget):
                if i != best_idx:
                    trial_solution = self.mutation_operator(self.population[i], best_solution)
                    if func(trial_solution) < fitness[i]:
                        self.population[i] = trial_solution
        return best_solution