import random
import numpy as np
from scipy.optimize import differential_evolution

class BBOB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func):
        return func(random.uniform(-5.0, 5.0))

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget):
    x = x0
    for _ in range(budget):
        x = func(x)
        if x < bounds[0]:
            x = bounds[0]
        elif x > bounds[1]:
            x = bounds[1]
        if random.random() < 0.4:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.2:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.4:
            x = random.uniform(bounds[0], bounds[1])
    return x

def bbo_optimize(func, x0, bounds, budget, dim):
    # Create a population of random individuals
    population = [x0] * 100
    for _ in range(100):  # Evolve the population for 100 generations
        population = differential_evolution(bbo_opt, [(bounds[0], bounds[1]), (bounds[0], bounds[1])], x0=population)

    # Evaluate the best individual in the population
    best_individual = population[np.argmax([func(individual) for individual in population])]
    best_individual_fitness = func(best_individual)

    return best_individual, best_individual_fitness

# Description: Black Box Optimization using BBOB
# Code: 