import numpy as np
import random
from scipy.optimize import differential_evolution

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        return self.fitnesses

    def mutate(self, individual):
        if random.random() < 0.2:
            dim = self.dim
            if random.random() < 0.5:
                dim -= 1
            if random.random() < 0.5:
                x = individual.copy()
                x[dim] = random.uniform(-5.0, 5.0)
                self.population[i] = x
        return individual

    def crossover(self, parent1, parent2):
        if random.random() < 0.5:
            dim = self.dim
            if random.random() < 0.5:
                dim -= 1
            if random.random() < 0.5:
                x = parent1.copy()
                x[dim] = parent2[dim]
                self.population[i] = x
            return parent1
        return parent1

# Initialize the algorithm
algorithm = NNEO(100, 5)

# Define a new function to evaluate
def func(x):
    return np.sin(x)

# Evaluate the function 100 times
for _ in range(100):
    algorithm()

# Select the best individual
best_individual = np.argmax(algorithm.fitnesses)

# Select a mutation strategy
def mutate_strategy(individual):
    return algorithm.mutate(individual)

# Select a crossover strategy
def crossover_strategy(parent1, parent2):
    return algorithm.crossover(parent1, parent2)

# Update the algorithm with the mutation and crossover strategies
algorithm = NNEO(100, 5)
algorithm.population = mutate_strategy(algorithm.population)
algorithm.population = crossover_strategy(algorithm.population, algorithm.population[0])

# Evaluate the function 100 times
for _ in range(100):
    algorithm()

# Select the best individual
best_individual = np.argmax(algorithm.fitnesses)

# Print the result
print("Best individual:", best_individual)
print("Best fitness:", algorithm.fitnesses[best_individual])