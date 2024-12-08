import numpy as np
import random

class MEEAM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.fitness_values = []
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Select the best points for the next generation
            next_generation = self.population[:int(self.population_size * 0.2)]

            # Perform crossover and mutation
            for i in range(self.population_size):
                if random.random() < self.crossover_rate:
                    parent1, parent2 = random.sample(next_generation, 2)
                    child = np.mean([parent1, parent2], axis=0) + np.random.uniform(-0.1, 0.1, self.dim)
                else:
                    child = next_generation[i]

                # Evaluate the fitness of the child
                fitness = func(child)
                self.fitness_values.append((child, fitness))

                # Replace the least fit individual with the child
                self.population[i] = child

            # Sort the population based on fitness
            self.fitness_values.sort(key=lambda x: x[1])

            # Select the best point in the population
            self.population = self.fitness_values[:self.population_size][:, 0]

        # Return the best point in the population
        return self.population[0]

# Example usage:
def func(x):
    return np.sum(x**2)

meeam = MEEAM(budget=100, dim=10)
best_point = meeam(func)
print(best_point)