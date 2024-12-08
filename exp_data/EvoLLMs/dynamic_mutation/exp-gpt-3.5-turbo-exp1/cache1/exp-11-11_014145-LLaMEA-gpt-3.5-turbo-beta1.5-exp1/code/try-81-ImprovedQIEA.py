import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(individual) for individual in self.population]
            sorted_indices = np.argsort(fitness)
            elite_index = sorted_indices[0]
            elite = self.population[elite_index]
            for i in range(self.budget):
                if i != elite_index:
                    mutation = 0.7 * self.population[i] + 0.3 * elite  # Diversified mutation strategy
                    quantum_mutation = 0.6 * self.population[i] + 0.4 * elite + np.random.choice([-1, 1]) * np.random.rand() * (elite - self.population[i])
                    self.population[i] = quantum_mutation if func(quantum_mutation) < fitness[i] else mutation
        best_index = np.argmin([func(individual) for individual in self.population])
        return self.population[best_index]