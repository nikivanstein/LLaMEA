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
        self.adaptive_mutation_rate = 0.1

    def __call__(self, func):
        # Initialize the population with random points
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.differential_evolution_population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        # Evaluate the fitness of each point in the population
        for i in range(self.population_size):
            fitness = func(self.population[i])
            self.fitness_values.append((self.population[i], fitness))

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
                    # Update the adaptive mutation rate
                    self.adaptive_mutation_rate = max(0.01, min(0.5, self.adaptive_mutation_rate + 0.05 * random.random()))
            else:
                child = next_generation[i][0]

            # Evaluate the fitness of the child
            fitness = func(child)
            self.fitness_values.append((child, fitness))

            # Update the differential evolution population
            self.differential_evolution_population[i] = child

        # Perform differential evolution
        for i in range(self.population_size):
            # Calculate the target vector
            target_vector = self.differential_evolution_population[i] + np.random.uniform(-0.1, 0.1, self.dim)

            # Find the closest individual
            closest_individual = np.argmin(np.linalg.norm(self.differential_evolution_population - target_vector, axis=1))
            closest_individual_vector = self.differential_evolution_population[closest_individual]

            # Update the individual
            self.population[i] = target_vector - 0.5 * (self.population[i] - closest_individual_vector)

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