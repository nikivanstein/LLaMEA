import numpy as np
import random

class CDEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crowd_size = 10
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.crowd = np.random.uniform(-5.0, 5.0, (self.crowd_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the population
            population_values = np.array([func(x) for x in self.population])

            # Evaluate the crowd
            crowd_values = np.array([func(x) for x in self.crowd])

            # Select the best individuals
            best_indices = np.argsort(population_values)[:, -self.crowd_size:]
            best_crowd_values = crowd_values[best_indices]

            # Select the worst individuals
            worst_indices = np.argsort(population_values)[:, :self.crowd_size]
            worst_population_values = population_values[worst_indices]

            # Update the population
            self.population = np.concatenate((best_crowd_values, worst_population_values))

            # Update the crowd
            self.crowd = self.population[:self.crowd_size]

            # Perform probability-based mutation
            mutated_population = self.mutate(self.population)
            # Randomly replace 20% of the population with mutated individuals
            mutated_indices = np.random.choice(len(self.population), size=len(self.population), replace=False, p=0.2)
            self.population[mutated_indices] = mutated_population[mutated_indices]

    def mutate(self, population):
        # Perform Gaussian mutation
        mutated_population = population + np.random.normal(0, 1, population.shape)
        return np.clip(mutated_population, -5.0, 5.0)

# Test the algorithm
def bbb(n, f):
    for i in range(n):
        func = f(i)
        cdea = CDEA(100, 5)
        cdea(func)
        print(f'Function {i+1}: {func()}')

# BBOB test suite of 24 noiseless functions
functions = [
    lambda x: sum(x**i for i in range(5)),
    lambda x: sum(x**i for i in range(3, 9)),
    lambda x: sum(x**i for i in range(2, 10)),
    lambda x: sum(x**i for i in range(1, 6)),
    lambda x: sum(x**i for i in range(0, 7)),
    lambda x: sum(x**i for i in range(-3, 4)),
    lambda x: sum(x**i for i in range(-2, 5)),
    lambda x: sum(x**i for i in range(-1, 6)),
    lambda x: sum(x**i for i in range(0, 6)),
    lambda x: sum(x**i for i in range(-3, 4)),
    lambda x: sum(x**i for i in range(-2, 5)),
    lambda x: sum(x**i for i in range(-1, 6)),
    lambda x: sum(x**i for i in range(0, 6)),
    lambda x: sum(x**i for i in range(-3, 4)),
    lambda x: sum(x**i for i in range(-2, 5)),
    lambda x: sum(x**i for i in range(-1, 6)),
    lambda x: sum(x**i for i in range(0, 6)),
    lambda x: sum(x**i for i in range(-3, 4)),
    lambda x: sum(x**i for i in range(-2, 5)),
    lambda x: sum(x**i for i in range(-1, 6)),
    lambda x: sum(x**i for i in range(0, 6)),
    lambda x: sum(x**i for i in range(-3, 4)),
    lambda x: sum(x**i for i in range(-2, 5)),
    lambda x: sum(x**i for i in range(-1, 6)),
]

for func in functions:
    bbb(len(functions), func)