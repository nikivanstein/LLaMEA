import numpy as np

class EnhancedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.mutation_rates = np.full(dim, 0.5)

    def evolve(self, fitness_values):
        sorted_indices = np.argsort(fitness_values)
        elite = self.population[sorted_indices[0]]
        success_rate = sum(fitness_values < fitness_values[0]) / self.budget
        
        self.mutation_rates = np.where(success_rate > 0.2, np.minimum(self.mutation_rates * 1.2, 2.0), np.maximum(self.mutation_rates * 0.8, 0.1))

        for i in range(1, self.budget):
            mutation_mask = np.random.rand(self.dim) < self.mutation_rates
            mutation_directions = np.where(np.random.rand(self.dim) < 0.5, 1, -1)
            self.population[i] += mutation_mask * mutation_directions * np.random.rand(self.dim) * (elite - self.population[i])

    def __call__(self, func):
        fitness_values = [func(ind) for ind in self.population]

        for _ in range(self.budget):
            self.evolve(fitness_values)
            fitness_values = [func(ind) for ind in self.population]

        best_solution = self.population[np.argmin(fitness_values)]
        return best_solution