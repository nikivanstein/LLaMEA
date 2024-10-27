import numpy as np
from scipy.optimize import differential_evolution
import random

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def mutate(self, individual):
        new_individual = individual.copy()
        if random.random() < 0.05:
            new_individual[0] = random.uniform(-5.0, 5.0)
        return new_individual

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        if random.random() < 0.5:
            child[1] = random.uniform(-5.0, 5.0)
        return child

# Example usage:
def func1(x):
    return x[0]**2 + x[1]**2

def func2(x):
    return np.sin(x[0]) + np.cos(x[1])

def func3(x):
    return x[0] + x[1]

algorithm = HEBBO(budget=100, dim=2)
individual = np.array([0, 0])
for _ in range(100):
    individual = algorithm(individual)
    new_individual = algorithm.mutate(individual)
    if new_individual[0] > 0:
        new_individual = algorithm.crossover(individual, new_individual)
    algorithm.func_evaluations += 1