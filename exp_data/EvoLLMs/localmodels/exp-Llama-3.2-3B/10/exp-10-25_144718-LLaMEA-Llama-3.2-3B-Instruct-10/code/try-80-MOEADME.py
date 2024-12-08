import numpy as np
import random
import time
from functools import lru_cache

class MOEADME:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.differential_evolution_rate = 0.1
        self.fitness_values = []
        self.differential_evolution_population = []
        self.differential_evolution_best = None

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

        # Select the best point in the population
        best_point = self.fitness_values[-1][0]

        # Perform differential evolution
        if random.random() < self.differential_evolution_rate:
            self.differential_evolution_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = random.sample(next_generation, 2)
                child = self.differential_evolution(parent1, parent2)
                self.differential_evolution_population.append(child)

            # Evaluate the fitness of the child
            fitness = func(child)
            self.fitness_values.append((child, fitness))

            # Update the best point
            if self.differential_evolution_best is None or fitness < self.differential_evolution_best[1]:
                self.differential_evolution_best = (child, fitness)

        # Return the best point in the population
        return best_point

    def differential_evolution(self, parent1, parent2):
        # Calculate the difference vector
        diff = parent2 - parent1

        # Calculate the scaling factor
        scaling_factor = np.random.uniform(0.5, 1.5)

        # Calculate the child
        child = parent1 + scaling_factor * diff

        # Return the child
        return child

# Example usage:
def func(x):
    return np.sum(x**2)

meeadme = MOEADME(budget=100, dim=10)
best_point = meeadme(func)
print(best_point)