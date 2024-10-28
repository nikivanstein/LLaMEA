import random
import numpy as np

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

    def __call__(self, func, x0, bounds, budget):
        return func(x0)

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget):
    population = [x0] * self.dim
    for _ in range(budget):
        new_population = []
        for individual in population:
            new_individual = func(individual, bounds)
            new_population.append(new_individual)
        population = new_population
    return population

def bbo_optimize(func, x0, bounds, budget, dim):
    # Novel Heuristic Algorithm: Evolutionary Black Box Optimization using Genetic Algorithm
    population = bbo_opt(func, x0, bounds, budget)
    # Refine the strategy by changing 20% of the individuals
    population = population[:int(0.2 * len(population))]
    for _ in range(dim):
        for individual in population:
            if random.random() < 0.4:
                individual = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.2:
                individual = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.4:
                individual = random.uniform(bounds[0], bounds[1])
    return population

# Description: Evolutionary Black Box Optimization using Genetic Algorithm
# Code: 