import numpy as np

class ImprovedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate fitness based on function value
            fitness = [func(individual) for individual in self.population]
            # Select parents based on fitness
            parents = self.population[np.argsort(fitness)[:2]]
            # Preserve top individuals
            top_idx = np.argsort(fitness)[:2]
            self.population = np.vstack([self.population[top_idx], self.population])
            # Perform adaptive mutation
            mutation_factor = 0.5 + 0.5 * np.exp(-_ / self.budget)
            offspring = mutation_factor * (parents[0] + parents[1]) + np.random.normal(0, 1, self.dim)
            # Replace worst individual with offspring
            worst_idx = np.argmax(fitness)
            self.population[worst_idx] = offspring
        return self.population[np.argmin([func(individual) for individual in self.population])]