import numpy as np
import random

class MEEAD:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.differential_evolution_rate = 0.1
        self.fitness_values = []
        self.differential_evolution_population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def __call__(self, func):
        # Initialize the population with random points
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        # Evaluate the fitness of each point in the population
        for i in range(self.population_size):
            fitness = func(population[i])
            self.fitness_values.append((population[i], fitness))

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

        # Perform differential evolution
        for i in range(self.population_size):
            if random.random() < self.differential_evolution_rate:
                differential_evolution = np.random.uniform(-5.0, 5.0, self.dim)
                differential_evolution = differential_evolution / np.linalg.norm(differential_evolution)
                differential_evolution = differential_evolution * np.random.uniform(0.5, 1.5)
                differential_evolution_population[i] += differential_evolution

        # Evaluate the fitness of the differential evolution population
        for i in range(self.population_size):
            fitness = func(differential_evolution_population[i])
            self.fitness_values.append((differential_evolution_population[i], fitness))

        # Sort the population based on fitness
        self.fitness_values.sort(key=lambda x: x[1])

        # Return the best point in the population
        return self.fitness_values[-1][0]

# Example usage:
def func(x):
    return np.sum(x**2)

meead = MEEAD(budget=100, dim=10)
best_point = meead(func)
print(best_point)