import numpy as np

class EnhancedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def __call__(self, func):
        mutation_rate = 0.1
        for _ in range(self.budget):
            fitness = [func(individual) for individual in self.population]
            sorted_indices = np.argsort(fitness)
            elite_index = sorted_indices[0]
            elite = self.population[elite_index]
            for i in range(self.budget):
                if i != elite_index:
                    mutation = self.population[i] + np.random.normal(0, mutation_rate, self.dim)
                    offspring = 0.5 * self.population[i] + 0.5 * elite
                    quantum_mutation = offspring + np.random.choice([-1, 1]) * np.random.rand() * (elite - self.population[i])
                    self.population[i] = quantum_mutation if func(quantum_mutation) < fitness[i] else mutation
            mutation_rate = mutation_rate * 0.995  # Adapt mutation rate
        best_index = np.argmin([func(individual) for individual in self.population])
        return self.population[best_index]