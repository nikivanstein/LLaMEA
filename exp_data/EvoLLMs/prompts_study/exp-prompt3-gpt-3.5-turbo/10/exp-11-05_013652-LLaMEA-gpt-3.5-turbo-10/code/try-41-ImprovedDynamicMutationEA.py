import numpy as np

class ImprovedDynamicMutationEA:
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
            if np.random.rand() < 0.1:  # 10% chance to adapt population size
                if np.random.rand() < 0.5 and len(self.population) < 2*self.budget:
                    self.population = np.vstack((self.population, np.random.uniform(-5.0, 5.0, (1, self.dim)))
                elif len(self.population) > self.budget:
                    to_remove = np.random.choice(len(self.population))
                    self.population = np.delete(self.population, to_remove, axis=0)
        return self.population[np.argmin([func(ind) for ind in self.population])]