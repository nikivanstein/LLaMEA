import numpy as np

class DynamicMutationEA:
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
            mutation = np.random.randn(self.dim) * mutation_rate
            new_solution = best_solution + mutation
            if func(new_solution) < fitness[best_idx]:
                self.population[best_idx] = new_solution
            # Dynamic population size adaptation
            if np.std(fitness) < 0.1: # If fitness diversity is low
                self.population = np.concatenate((self.population, np.random.uniform(-5.0, 5.0, (self.budget, self.dim))))
            elif np.std(fitness) > 1.0: # If fitness diversity is high
                self.population = self.population[:self.budget]
        return self.population[np.argmin([func(ind) for ind in self.population])]