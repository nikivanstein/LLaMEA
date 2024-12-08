import numpy as np
import random

class MEEADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.fitness_values = []
        self.diff_evolution_rate = 0.1
        self.diff_evolution_count = 10

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
        for _ in range(self.diff_evolution_count):
            if random.random() < self.diff_evolution_rate:
                # Select two random points from the population
                parent1, parent2 = random.sample(self.fitness_values, 2)
                # Calculate the difference vector
                diff_vector = parent1[0] - parent2[0]
                # Calculate the target vector
                target_vector = parent1[0] + diff_vector * np.random.uniform(-1, 1, self.dim)
                # Calculate the new point
                new_point = target_vector + parent2[0] * np.random.uniform(-1, 1, self.dim)
                # Evaluate the fitness of the new point
                fitness = func(new_point)
                self.fitness_values.append((new_point, fitness))

        # Return the best point in the population
        return self.fitness_values[-1][0]

# Example usage:
def func(x):
    return np.sum(x**2)

meeade = MEEADE(budget=100, dim=10)
best_point = meead(func)
print(best_point)