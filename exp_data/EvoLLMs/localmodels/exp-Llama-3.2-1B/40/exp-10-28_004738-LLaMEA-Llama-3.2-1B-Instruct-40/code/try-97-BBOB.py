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
            func = lambda x: np.random.uniform(-5.0, 5.0, self.dim)
            functions.append(func)
        return functions

    def __call__(self, func):
        return func(np.random.uniform(-5.0, 5.0, self.dim))

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def f_double_prime_prime_prime(x):
    return 8

def bbo_opt(func, x0, bounds, budget):
    population_size = 100
    population = [x0] * population_size
    for _ in range(budget):
        for i in range(population_size):
            x = population[i]
            if random.random() < 0.4:
                x = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.2:
                x = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.4:
                x = random.uniform(bounds[0], bounds[1])
            population[i] = func(x)
    return population

def bbo_bounded_opt(func, x0, bounds, budget, bounds_size):
    population_size = 100
    population = [x0] * population_size
    for _ in range(budget):
        for i in range(population_size):
            x = population[i]
            if random.random() < 0.4:
                x = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.2:
                x = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.4:
                x = random.uniform(bounds[0], bounds[1])
            population[i] = func(x)
    return population

def bbo_bounded_opt_prime(func, x0, bounds, budget, bounds_size):
    population_size = 100
    population = [x0] * population_size
    for _ in range(budget):
        for i in range(population_size):
            x = population[i]
            if random.random() < 0.4:
                x = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.2:
                x = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.4:
                x = random.uniform(bounds[0], bounds[1])
            population[i] = func(x)
    return population

# One-line description: Evolutionary Algorithm for Multi-Dimensional Optimization
# Code: 
# ```python
# BBOB Algorithm
# 
# The BBOB algorithm is a novel metaheuristic for solving black box optimization problems. It uses a population-based approach, where each individual in the population is a candidate solution, and the population evolves over iterations using a combination of random perturbations and bounds.