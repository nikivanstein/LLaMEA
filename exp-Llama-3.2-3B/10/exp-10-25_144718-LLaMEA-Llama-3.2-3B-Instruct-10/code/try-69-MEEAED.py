import numpy as np
import random

class MEEAED:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.fitness_values = []
        self.differential_evolution_parameters = {
            'pop_size': 50,
            'cr': 0.5,
           'maxiter': 100
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
        for _ in range(self.differential_evolution_parameters['maxiter']):
            for i in range(self.population_size):
                # Calculate the target vector
                target_vector = np.mean(next_generation, axis=0)

                # Calculate the trial vector
                trial_vector = target_vector + np.random.uniform(-1, 1, self.dim)

                # Calculate the difference vector
                difference_vector = trial_vector - next_generation[i]

                # Calculate the scaled difference vector
                scaled_difference_vector = difference_vector / np.linalg.norm(difference_vector)

                # Perform crossover
                child = next_generation[i] + self.crossover_rate * scaled_difference_vector

                # Perform mutation
                if random.random() < self.mutation_rate:
                    child += np.random.uniform(-0.1, 0.1, self.dim)

                # Evaluate the fitness of the child
                fitness = func(child)
                self.fitness_values.append((child, fitness))

        # Sort the population based on fitness
        self.fitness_values.sort(key=lambda x: x[1])

        # Return the best point in the population
        return self.fitness_values[-1][0]

# Example usage:
def func(x):
    return np.sum(x**2)

meeaed = MEEAED(budget=100, dim=10)
best_point = meaed(func)
print(best_point)