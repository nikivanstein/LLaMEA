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
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness = np.zeros((self.population_size,))

    def __call__(self, func):
        start_time = time.time()
        for _ in range(self.budget):
            # Evaluate the fitness of each point in the population
            self.fitness = func(self.population)
            self.fitness_values = np.array(list(zip(self.population, self.fitness)))

            # Sort the population based on fitness
            self.fitness_values.sort(key=lambda x: x[1], axis=0)

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
                self.fitness_values = np.vstack((self.fitness_values, (child, fitness)))

                # Update the population
                self.population[i] = child

            # Update the fitness values
            self.fitness_values.sort(key=lambda x: x[1])

        # Return the best point in the population
        best_point = self.fitness_values[-1][0]
        end_time = time.time()
        print(f"Best point: {best_point}, Time taken: {end_time - start_time} seconds")
        return best_point

# Example usage:
def func(x):
    return np.sum(x**2)

meeam = MEEAM(budget=100, dim=10)
best_point = meeam(func)
print(best_point)