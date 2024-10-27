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
            # Select the best points for the next generation
            next_generation = self.fitness_values[:int(self.population_size * 0.2)]

            # Perform crossover and mutation
            new_population = []
            for i in range(self.population_size):
                if random.random() < self.crossover_rate:
                    parent1, parent2 = random.sample(next_generation, 2)
                    child = np.mean([parent1[0], parent2[0]], axis=0)
                    if random.random() < self.probability:
                        child += np.random.uniform(-0.1, 0.1, self.dim)
                    new_population.append(child)
                else:
                    new_population.append(next_generation[i][0])

            # Evaluate the fitness of the new population
            for individual in new_population:
                fitness = func(individual)
                self.fitness_values.append((individual, fitness))

            # Update the population
            self.population = new_population

            # Sort the population based on fitness
            self.fitness_values.sort(key=lambda x: x[1])

            # Select the best points for the next generation
            next_generation = self.fitness_values[:int(self.population_size * 0.2)]

        # Return the best point in the population
        return self.fitness_values[-1][0]

# Example usage:
def func(x):
    return np.sum(x**2)

meeam = MEEAM(budget=100, dim=10)
best_point = meeam(func)
print(best_point)