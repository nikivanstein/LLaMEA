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
            # Select individuals for crossover and mutation
            selected_individuals = random.sample(self.population, int(self.population_size * 0.2))

            # Perform crossover and mutation
            new_individuals = []
            for individual in selected_individuals:
                if random.random() < self.probability:
                    parent1, parent2 = random.sample(selected_individuals, 2)
                    child = np.mean([parent1, parent2], axis=0)
                    if random.random() < self.mutation_rate:
                        child += np.random.uniform(-0.1, 0.1, self.dim)
                else:
                    child = individual

                # Evaluate the fitness of the child
                fitness = func(child)
                self.fitness_values.append((child, fitness))
                new_individuals.append(child)

            # Update the population
            self.population = np.array(new_individuals)

            # Sort the population based on fitness
            self.fitness_values.sort(key=lambda x: x[1])

            # Select the best points for the next generation
            self.population = self.population[:int(self.population_size * 0.2)]

# Example usage:
def func(x):
    return np.sum(x**2)

meeam = MEEAM(budget=100, dim=10)
best_point = meeam(func)
print(best_point)