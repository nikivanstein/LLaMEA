import numpy as np
import random

class MOEAM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.probability = 0.1
        self.fitness_values = []

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
                    if random.random() < self.probability:
                        child = self.adaptive_mutation(child)
                child = self.probabilistic_crossover(child, parent1[0], parent2[0])
            else:
                child = next_generation[i][0]

            # Evaluate the fitness of the child
            fitness = func(child)
            self.fitness_values.append((child, fitness))

        # Sort the population based on fitness
        self.fitness_values.sort(key=lambda x: x[1])

        # Return the best point in the population
        return self.fitness_values[-1][0]

    def adaptive_mutation(self, child):
        # Adapt the mutation based on the fitness of the child
        if self.fitness_values[-1][1] > self.fitness_values[int(len(self.fitness_values) / 2)][1]:
            mutation = np.random.uniform(-0.1, 0.1, self.dim)
        else:
            mutation = np.zeros(self.dim)
        child += mutation
        return child

    def probabilistic_crossover(self, child, parent1, parent2):
        # Perform probabilistic crossover between the child and the parents
        if random.random() < self.probability:
            crossover_point = random.randint(0, self.dim - 1)
            child[crossover_point:] = parent2[crossover_point:]
        return child

# Example usage:
def func(x):
    return np.sum(x**2)

meeam = MOEAM(budget=100, dim=10)
best_point = meeam(func)
print(best_point)