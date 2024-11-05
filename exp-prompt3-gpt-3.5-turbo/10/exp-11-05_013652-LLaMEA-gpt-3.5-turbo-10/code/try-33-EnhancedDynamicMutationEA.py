import numpy as np

class EnhancedDynamicMutationEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_solution = self.population[best_idx]
            mutation_rate = np.random.uniform(0.01, 0.1)
            for idx, ind in enumerate(self.population):
                mutation_rate = 0.01 + 0.09 * (1.0 - (func(ind) - np.min(fitness)) / (np.max(fitness) - np.min(fitness)))
                mutation = np.random.randn(self.dim) * mutation_rate
                new_solution = ind + mutation
                if func(new_solution) < fitness[idx]:
                    self.population[idx] = new_solution
        return self.population[np.argmin([func(ind) for ind in self.population])]