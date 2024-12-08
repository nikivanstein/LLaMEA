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

    def __call__(self, func):
        for _ in range(self.budget):
            # Initialize the population with random points
            population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

            # Evaluate the fitness of each point in the population
            fitness_values = []
            for i in range(self.population_size):
                fitness = func(population[i])
                fitness_values.append(fitness)

            # Sort the population based on fitness
            sorted_indices = np.argsort(fitness_values)
            population = population[sorted_indices]

            # Select the best points for the next generation
            next_generation = population[:int(self.population_size * 0.2)]

            # Perform crossover and mutation
            for i in range(self.population_size):
                if random.random() < self.crossover_rate:
                    parent1, parent2 = random.sample(next_generation, 2)
                    child = np.mean([parent1, parent2], axis=0)
                    if random.random() < self.mutation_rate:
                        child += np.random.uniform(-0.1, 0.1, self.dim)
                else:
                    child = next_generation[i]

                # Evaluate the fitness of the child
                fitness = func(child)
                fitness_values.append(fitness)

            # Sort the population based on fitness
            sorted_indices = np.argsort(fitness_values)
            population = population[sorted_indices]

            # Update the best point
            self.fitness_values.append((population[0], fitness_values[0]))

        # Return the best point in the population
        return self.fitness_values[-1][0]

# Example usage:
def func(x):
    return np.sum(x**2)

meeam = MEEAM(budget=100, dim=10)
best_point = meeam(func)
print(best_point)