import numpy as np
import random

class MEEAMDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.differential_evolution_rate = 0.2
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness_values = []
        self.differential_evolution_population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the fitness of each point in the population
            self.fitness_values = []
            for i in range(self.population_size):
                fitness = func(self.population[i])
                self.fitness_values.append((self.population[i], fitness))

            # Sort the population based on fitness
            self.fitness_values.sort(key=lambda x: x[1])

            # Select the best points for the next generation
            next_generation = self.fitness_values[:int(self.population_size * 0.2)]

            # Perform crossover and mutation
            for i in range(self.population_size):
                if random.random() < self.crossover_rate:
                    parent1, parent2 = random.sample(next_generation, 2)
                    child = np.mean([parent1[0], parent2[0]], axis=0)
                    if random.random() < self.mutation_rate:
                        child += np.random.uniform(-0.1, 0.1, self.dim)
                else:
                    child = next_generation[i][0]

                # Evaluate the fitness of the child
                fitness = func(child)
                self.fitness_values.append((child, fitness))

            # Sort the population based on fitness
            self.fitness_values.sort(key=lambda x: x[1])

            # Update the population
            self.population = next_generation

            # Perform differential evolution
            if random.random() < self.differential_evolution_rate:
                for i in range(self.population_size):
                    # Calculate the differential evolution parameters
                    x1, x2 = random.sample(self.population, 2)
                    x3, x4 = random.sample(self.population, 2)

                    # Calculate the differential evolution vector
                    diff = x2 - x1
                    diff = diff / np.linalg.norm(diff)

                    # Update the individual
                    self.differential_evolution_population[i] = x1 + 0.5 * diff * (x3 - x4)

                    # Evaluate the fitness of the individual
                    fitness = func(self.differential_evolution_population[i])
                    self.fitness_values.append((self.differential_evolution_population[i], fitness))

                # Update the population
                self.population = np.concatenate((self.population, self.differential_evolution_population))

        # Return the best point in the population
        return self.fitness_values[-1][0]

# Example usage:
def func(x):
    return np.sum(x**2)

meeamde = MEEAMDE(budget=100, dim=10)
best_point = meeamde(func)
print(best_point)