import numpy as np
import random

class EvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.swarm_size = 10
        self.probability = 0.25

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        for _ in range(self.budget):
            fitness = np.array([func(x) for x in population])
            fittest = population[np.argsort(fitness)]
            offspring = []
            for _ in range(self.population_size):
                if random.random() < self.crossover_rate:
                    parent1, parent2 = random.sample(fittest, 2)
                    child = np.concatenate((parent1[:self.dim//2], parent2[self.dim//2:]))
                    if random.random() < self.probability:
                        child += np.random.uniform(-0.1, 0.1, self.dim)
                else:
                    child = parent1
                offspring.append(child)
            population = np.array(offspring)
        return population[np.argmin(fitness)]

# Example usage
def func(x):
    return np.sum(x**2)

optimizer = EvolutionaryOptimizer(budget=100, dim=5)
best_solution = optimizer(func)
print("Best solution:", best_solution)