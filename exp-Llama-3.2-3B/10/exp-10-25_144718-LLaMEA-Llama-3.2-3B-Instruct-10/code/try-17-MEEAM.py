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
            # Evaluate the fitness of each point in the population
            fitness_values = []
            for individual in self.population:
                fitness = func(individual)
                fitness_values.append(fitness)
            self.fitness_values = zip(self.population, fitness_values)

            # Sort the population based on fitness
            self.population = sorted(self.population, key=lambda x: x[1], reverse=True)

            # Select the best points for the next generation
            next_generation = self.population[:int(self.population_size * 0.2)]

            # Perform crossover and mutation
            for i in range(self.population_size - len(next_generation)):
                if random.random() < self.crossover_rate:
                    parent1, parent2 = random.sample(next_generation, 2)
                    child = np.mean([parent1[0], parent2[0]], axis=0)
                    if random.random() < self.mutation_rate:
                        child += np.random.uniform(-0.1, 0.1, self.dim)
                else:
                    child = next_generation[i][0]

                # Evaluate the fitness of the child
                fitness = func(child)
                fitness_values.append(fitness)

            # Update the population
            self.population = next_generation + [child for child, _ in self.fitness_values[len(next_generation):]]

            # Change the individual lines with 0.1 probability
            if random.random() < 0.1:
                for i in range(self.population_size):
                    if random.random() < 0.5:
                        self.population[i] += np.random.uniform(-0.1, 0.1, self.dim)

        # Return the best point in the population
        return self.population[0]

# Example usage:
def func(x):
    return np.sum(x**2)

meeam = MEEAM(budget=100, dim=10)
best_point = meeam(func)
print(best_point)