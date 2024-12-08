import numpy as np

class AdaptiveMutationEA:
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
                if idx != best_idx:
                    dynamic_mutation_rate = mutation_rate * (1 / (1 + np.abs(func(ind) - func(best_solution))))
                    mutation = np.random.randn(self.dim) * dynamic_mutation_rate
                    new_solution = ind + mutation
                    if func(new_solution) < fitness[idx]:
                        self.population[idx] = new_solution
        return self.population[np.argmin([func(ind) for ind in self.population])]