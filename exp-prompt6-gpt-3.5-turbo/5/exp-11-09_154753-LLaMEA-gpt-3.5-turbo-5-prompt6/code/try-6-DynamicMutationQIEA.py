import numpy as np

class DynamicMutationQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
    
    def __call__(self, func):
        mutation_rate = 0.1  # Initial mutation rate
        for _ in range(self.budget):
            fitness = [func(individual) for individual in self.population]
            parents = self.population[np.argsort(fitness)[:2]]
            offspring = 0.5 * (parents[0] + parents[1]) + np.random.normal(0, mutation_rate, self.dim)
            worst_idx = np.argmax(fitness)
            self.population[worst_idx] = offspring
            # Adjust mutation rate based on fitness
            mutation_rate *= np.exp(-0.01 * (fitness[worst_idx] - np.mean(fitness)))
        return self.population[np.argmin([func(individual) for individual in self.population])]