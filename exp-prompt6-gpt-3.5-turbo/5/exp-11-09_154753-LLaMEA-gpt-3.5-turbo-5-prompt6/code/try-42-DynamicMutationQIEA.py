import numpy as np

class DynamicMutationQIEA:
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
            # Dynamic mutation rate based on individual fitness
            mutation_rate = 0.1 + 0.4 * (1 - np.mean(fitness) / np.max(fitness))
            # Perform crossover and mutation with dynamic mutation rate
            offspring = 0.5 * (parents[0] + parents[1]) + mutation_rate * np.random.normal(0, 1, self.dim)
            # Replace worst individual with offspring
            worst_idx = np.argmax(fitness)
            self.population[worst_idx] = offspring
        return self.population[np.argmin([func(individual) for individual in self.population])]