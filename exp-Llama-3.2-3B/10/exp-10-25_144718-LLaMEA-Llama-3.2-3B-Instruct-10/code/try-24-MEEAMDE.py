import numpy as np
import random

class MEEAMDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.fitness_values = []
        self.differential_evolution_params = {
            'F': 0.5,  # Full search
            'CR': 0.5,  # Crossover rate
            'AT': 2.0  # Acceptance threshold
        }

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

        # Perform differential evolution
        for i in range(self.population_size):
            # Randomly select three individuals
            x1, x2, x3 = random.sample(next_generation, 3)

            # Calculate the differential vector
            F = self.differential_evolution_params['F']
            d = x3 - x1
            r1 = random.random()
            r2 = random.random()
            if r1 < 0.5:
                d = -d
            if r2 < 0.5:
                d = -d

            # Calculate the trial point
            x = x1 + F * d
            if random.random() < self.mutation_rate:
                x += np.random.uniform(-0.1, 0.1, self.dim)

            # Evaluate the fitness of the trial point
            fitness = func(x)
            self.fitness_values.append((x, fitness))

        # Sort the population based on fitness
        self.fitness_values.sort(key=lambda x: x[1])

        # Return the best point in the population
        return self.fitness_values[-1][0]

# Example usage:
def func(x):
    return np.sum(x**2)

meeamde = MEEAMDE(budget=100, dim=10)
best_point = meeamde(func)
print(best_point)