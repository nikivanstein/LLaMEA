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
        self.differential_evolution_population = []
        self.best_individual = None

    def __call__(self, func):
        # Initialize the population with random points
        self.differential_evolution_population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        # Evaluate the fitness of each point in the population
        for i in range(self.population_size):
            fitness = func(self.differential_evolution_population[i])
            self.fitness_values.append((self.differential_evolution_population[i], fitness))

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
            if self.best_individual is None or self.fitness_values[self.fitness_values.index(self.best_individual)][1] > self.fitness_values[i][1]:
                self.best_individual = self.fitness_values[i][0]

            # Calculate the differential evolution parameters
            rand1 = random.random()
            rand2 = random.random()
            rand3 = random.random()
            rand4 = random.random()

            # Calculate the differential evolution vector
            x = self.differential_evolution_population[i]
            y = self.differential_evolution_population[rand1 * self.population_size + rand2 % self.population_size]
            z = self.differential_evolution_population[rand3 * self.population_size + rand4 % self.population_size]
            d = self.fitness_values.index((x, self.fitness_values[i][1])) - self.fitness_values.index((y, self.fitness_values[i][1])) - self.fitness_values.index((z, self.fitness_values[i][1]))
            dx = (x - y) / abs(d)
            dy = (y - z) / abs(d)
            dz = (z - x) / abs(d)

            # Perform differential evolution
            child = x + rand1 * dx + rand2 * dy + rand3 * dz
            if random.random() < self.mutation_rate:
                child += np.random.uniform(-0.1, 0.1, self.dim)

            # Evaluate the fitness of the child
            fitness = func(child)
            self.fitness_values.append((child, fitness))

        # Sort the population based on fitness
        self.fitness_values.sort(key=lambda x: x[1])

        # Return the best point in the population
        return self.best_individual

# Example usage:
def func(x):
    return np.sum(x**2)

meeamde = MEEAMDE(budget=100, dim=10)
best_point = meeamde(func)
print(best_point)