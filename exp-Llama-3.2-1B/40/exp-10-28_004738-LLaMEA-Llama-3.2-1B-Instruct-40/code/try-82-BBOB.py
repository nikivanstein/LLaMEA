import random
import math

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
    population = [x0] * budget
    for _ in range(budget):
        new_population = []
        for i in range(budget):
            new_individual = population[i]
            # Adaptive mutation
            if random.random() < 0.4:
                new_individual = random.uniform(bounds[0], bounds[1])
            # Adaptive selection
            if random.random() < 0.2:
                new_individual = random.uniform(bounds[0], bounds[1])
            # Evaluate fitness
            fitness = func(new_individual)
            new_population.append(new_individual)
            new_population[i] = fitness
        population = new_population
    return population

# Evolutionary Algorithm with Adaptive Mutation and Selection
# Code: 