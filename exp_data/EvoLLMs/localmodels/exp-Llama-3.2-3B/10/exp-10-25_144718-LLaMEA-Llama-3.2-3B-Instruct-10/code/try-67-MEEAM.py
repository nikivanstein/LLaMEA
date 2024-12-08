import numpy as np
import random
import time

class MEEAM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.fitness_values = []
        self.start_time = time.time()

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

    def refine_strategy(self):
        # Refine the strategy by changing the individual lines
        for i in range(self.population_size):
            if random.random() < 0.1:
                # Change the mutation rate
                self.mutation_rate = np.random.uniform(0.0, 0.2)
                # Change the crossover rate
                self.crossover_rate = np.random.uniform(0.0, 0.2)
                # Change the population size
                self.population_size = np.random.randint(20, 100)

        # Evaluate the fitness of each point in the population
        for i in range(self.population_size):
            fitness = func(self.fitness_values[i][0])
            self.fitness_values[i] = (self.fitness_values[i][0], fitness)

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