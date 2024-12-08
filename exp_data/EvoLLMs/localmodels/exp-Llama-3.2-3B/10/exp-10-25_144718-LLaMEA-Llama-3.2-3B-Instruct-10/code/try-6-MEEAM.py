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

        # Return the best point in the population
        return self.fitness_values[-1][0]

    def adjust_crossover_probability(self):
        # Adjust the crossover probability based on the population's diversity
        population = np.array([x[0] for x in self.fitness_values])
        distances = np.linalg.norm(population[:, np.newaxis] - population, axis=2)
        average_distance = np.mean(distances)
        self.crossover_rate = 0.5 * (1 + average_distance / 10)

    def adjust_mutation_probability(self):
        # Adjust the mutation probability based on the population's fitness
        fitness = [x[1] for x in self.fitness_values]
        average_fitness = np.mean(fitness)
        self.mutation_rate = 0.1 * (1 + (1 - average_fitness) / 10)

# Example usage:
def func(x):
    return np.sum(x**2)

meeam = MEEAM(budget=100, dim=10)
for _ in range(100):
    meeam.adjust_crossover_probability()
    meeam.adjust_mutation_probability()
    best_point = meeam(func)
    print(best_point)