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
        self.differential_evolution_params = {
            'pop_size': 20,
            'cr': 0.5,
           'maxiter': 100,
            'x0': np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
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
        for _ in range(self.differential_evolution_params['maxiter']):
            for i in range(self.differential_evolution_params['pop_size']):
                # Calculate the target vector
                target = self.fitness_values[i][0] + self.fitness_values[i][0] * self.fitness_values[i][1] * np.random.uniform(-1, 1, self.dim)

                # Calculate the trial vector
                trial = self.fitness_values[i][0] + self.fitness_values[i][0] * self.fitness_values[i][1] * np.random.uniform(-1, 1, self.dim)

                # Calculate the difference vector
                diff = target - trial

                # Calculate the new individual
                new_individual = trial + diff * self.mutation_rate

                # Evaluate the fitness of the new individual
                fitness = func(new_individual)
                self.fitness_values.append((new_individual, fitness))

                # Replace the worst individual with the new individual
                self.fitness_values.sort(key=lambda x: x[1])
                if self.fitness_values[-1][1] > self.fitness_values[i][1]:
                    self.fitness_values[i] = self.fitness_values[-1]

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

# Example usage:
def func(x):
    return np.sum(x**2)

meeam = MEEAM(budget=100, dim=10)
best_point = meeam(func)
print(best_point)