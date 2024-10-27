import numpy as np
import random
import copy

class MEEAM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.fitness_values = []

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
                child = np.clip(child, -5.0, 5.0)  # Clip the values to the search space
            else:
                child = next_generation[i][0]

            # Evaluate the fitness of the child
            fitness = func(child)
            self.fitness_values.append((child, fitness))

        # Sort the population based on fitness
        self.fitness_values.sort(key=lambda x: x[1])

        # Return the best point in the population
        return self.fitness_values[-1][0]

def adaptive_meeam(func, budget, dim):
    # Initialize the population with random points
    population = np.random.uniform(-5.0, 5.0, (budget, dim))

    # Evaluate the fitness of each point in the population
    for i in range(budget):
        fitness = func(population[i])
        population[i] = (population[i], fitness)

    # Sort the population based on fitness
    population.sort(key=lambda x: x[1])

    # Select the best points for the next generation
    next_generation = population[:int(budget * 0.2)]

    # Perform crossover and mutation
    for i in range(budget):
        if random.random() < 0.1:
            parent1, parent2 = random.sample(next_generation, 2)
            child = np.mean([parent1[0], parent2[0]], axis=0)
            if random.random() < 0.1:
                child += np.random.uniform(-0.1, 0.1, dim)
            child = np.clip(child, -5.0, 5.0)  # Clip the values to the search space
            child = (child, func(child))
        else:
            child = next_generation[i]
        next_generation[i] = child

    # Sort the population based on fitness
    next_generation.sort(key=lambda x: x[1])

    # Return the best point in the population
    return next_generation[-1][0]

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
best_point = adaptive_meeam(func, budget, dim)
print(best_point)