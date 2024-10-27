import numpy as np
import random

class MEEAM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.probability = 0.1
        self.fitness_values = []
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Select parents for crossover
            parents = random.sample(self.population, 2)

            # Perform crossover
            child = np.mean([parents[0][0], parents[1][0]], axis=0)
            if random.random() < self.probability:
                child += np.random.uniform(-0.1, 0.1, self.dim)

            # Evaluate the fitness of the child
            fitness = func(child)
            self.fitness_values.append((child, fitness))

            # Replace the worst individual in the population with the child
            self.population[np.argmin([x[1] for x in self.fitness_values])], child = child, fitness
            self.fitness_values.sort(key=lambda x: x[1])

        # Return the best point in the population
        return self.population[-1][0]

# Example usage:
def func(x):
    return np.sum(x**2)

meeam = MEEAM(budget=100, dim=10)
best_point = meeam(func)
print(best_point)