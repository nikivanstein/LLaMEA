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
            # Introducing Levy flight for exploration
            levy_flight = np.random.standard_cauchy(self.dim) / (np.sqrt(np.arange(1, self.dim + 1)))
            new_solution = best_solution + mutation_rate * levy_flight
            if func(new_solution) < fitness[best_idx]:
                self.population[best_idx] = new_solution
        return self.population[np.argmin([func(ind) for ind in self.population])]