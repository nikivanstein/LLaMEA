import numpy as np

class HybridDELS:
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
            # Introducing Local Search for exploitation
            for idx, ind in enumerate(self.population):
                candidate = ind + np.random.uniform(-0.1, 0.1, self.dim)
                if func(candidate) < fitness[idx]:
                    self.population[idx] = candidate
        return self.population[np.argmin([func(ind) for ind in self.population])]