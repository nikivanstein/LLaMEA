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
            new_population = np.concatenate((best_crowd_values, worst_population_values))
            new_population = np.random.choice(new_population, size=self.population_size, replace=True, p=0.5)
            self.population = new_population

            # Update the crowd
            self.crowd = self.population[:self.crowd_size]

            # Perform crossover and mutation
            self.population = self.crossover(self.population)
            self.population = self.mutate(self.population)

    def crossover(self, population):
        # Perform single-point crossover
        offspring = []
        for _ in range(len(population)):
            parent1, parent2 = random.sample(population, 2)
            child = np.concatenate((parent1, parent2[1:]))
            offspring.append(child)
        return np.array(offspring)

    def mutate(self, population):
        # Perform Gaussian mutation
        mutated_population = population + np.random.normal(0, 1, population.shape)
        return np.clip(mutated_population, -5.0, 5.0)

# Example usage:
# BBOB test suite
from bbopt import BBOPT

# Define the functions to optimize
def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return x[0]**2 * x[1]**2

def f3(x):
    return np.sin(x[0]) * np.cos(x[1])

def f4(x):
    return np.sin(x[0]) * np.sin(x[1])

def f5(x):
    return np.exp(x[0]) * np.exp(x[1])

def f6(x):
    return np.exp(x[0]) * np.exp(x[1]) + x[0]**2 + x[1]**2

def f7(x):
    return np.exp(x[0]) * np.exp(x[1]) + x[0]**2 * x[1]**2

def f8(x):
    return np.exp(x[0]) * np.exp(x[1]) + np.sin(x[0]) * np.cos(x[1])

def f9(x):
    return np.exp(x[0]) * np.exp(x[1]) + np.sin(x[0]) * np.sin(x[1])

def f10(x):
    return np.exp(x[0]) * np.exp(x[1]) + np.sin(x[0]) * np.cos(x[1]) + x[0]**2 + x[1]**2

def f11(x):
    return np.exp(x[0]) * np.exp(x[1]) + np.sin(x[0]) * np.sin(x[1]) + x[0]**2 * x[1]**2

def f12(x):
    return np.exp(x[0]) * np.exp(x[1]) + np.sin(x[0]) * np.cos(x[1]) + x[0]**2 * x[1]**2

def f13(x):
    return np.exp(x[0]) * np.exp(x[1]) + np.sin(x[0]) * np.sin(x[1]) + x[0]**2 + x[1]**2

def f14(x):
    return np.exp(x[0]) * np.exp(x[1]) + np.sin(x[0]) * np.cos(x[1]) + x[0]**2 * x[1]**2

def f15(x):
    return np.exp(x[0]) * np.exp(x[1]) + np.sin(x[0]) * np.sin(x[1]) + x[0]**2 * x[1]**2

def f16(x):
    return np.exp(x[0]) * np.exp(x[1]) + np.sin(x[0]) * np.cos(x[1]) + x[0]**2 + x[1]**2

def f17(x):
    return np.exp(x[0]) * np.exp(x[1]) + np.sin(x[0]) * np.sin(x[1]) + x[0]**2 * x[1]**2

def f18(x):
    return np.exp(x[0]) * np.exp(x[1]) + np.sin(x[0]) * np.cos(x[1]) + x[0]**2 + x[1]**2

def f19(x):
    return np.exp(x[0]) * np.exp(x[1]) + np.sin(x[0]) * np.sin(x[1]) + x[0]**2 * x[1]**2

def f20(x):
    return np.exp(x[0]) * np.exp(x[1]) + np.sin(x[0]) * np.cos(x[1]) + x[0]**2 + x[1]**2

def f21(x):
    return np.exp(x[0]) * np.exp(x[1]) + np.sin(x[0]) * np.sin(x[1]) + x[0]**2 * x[1]**2

def f22(x):
    return np.exp(x[0]) * np.exp(x[1]) + np.sin(x[0]) * np.cos(x[1]) + x[0]**2 + x[1]**2

def f23(x):
    return np.exp(x[0]) * np.exp(x[1]) + np.sin(x[0]) * np.sin(x[1]) + x[0]**2 * x[1]**2

def f24(x):
    return np.exp(x[0]) * np.exp(x[1]) + np.sin(x[0]) * np.cos(x[1]) + x[0]**2 + x[1]**2

# Create an instance of the CDEA algorithm
cdea = CDEA(budget=100, dim=2)

# Define the function to optimize
def func(x):
    return f24(x)

# Evaluate the function using the CDEA algorithm
cdea(func)

# Print the optimized values
print(cdea.population)