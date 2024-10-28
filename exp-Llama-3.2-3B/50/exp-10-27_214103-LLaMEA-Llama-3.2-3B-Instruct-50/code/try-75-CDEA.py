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

            # Perform probabilistic crossover and mutation
            self.population = self.crossover(self.population)
            self.population = self.mutate(self.population)

    def crossover(self, population):
        # Perform single-point crossover with probability 0.5
        offspring = []
        for _ in range(len(population)):
            if random.random() < 0.5:
                parent1, parent2 = random.sample(population, 2)
                child = np.concatenate((parent1, parent2[1:]))
            else:
                child = parent1
            offspring.append(child)
        return np.array(offspring)

    def mutate(self, population):
        # Perform Gaussian mutation with probability 0.5
        mutated_population = population + np.random.normal(0, 1, population.shape)
        mutated_population = np.clip(mutated_population, -5.0, 5.0)
        return mutated_population

# Example usage:
def bbb_1d(x):
    return x**2

def bbb_2d(x):
    return x[0]**2 + x[1]**2

def bbb_3d(x):
    return x[0]**2 + x[1]**2 + x[2]**2

def bbb_4d(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2

def bbb_5d(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2

def bbb_6d(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2

def bbb_7d(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2

def bbb_8d(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2

def bbb_9d(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2

def bbb_10d(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2

# Test the algorithm on the BBOB test suite
budget = 100
dim = 10
cdea = CDEA(budget, dim)
for func in [bbb_1d, bbb_2d, bbb_3d, bbb_4d, bbb_5d, bbb_6d, bbb_7d, bbb_8d, bbb_9d, bbb_10d]:
    cdea(func)