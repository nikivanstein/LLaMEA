import random
import numpy as np

class AdaptiveBBOOpt:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()
        self.population = self.initialize_population()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def initialize_population(self):
        return [np.random.uniform(-5.0, 5.0) for _ in range(self.dim)]

    def __call__(self, func):
        return func(random.uniform(-5.0, 5.0))

    def mutate(self, individual):
        if random.random() < 0.5:
            return individual + random.uniform(-1.0, 1.0)
        elif random.random() < 0.2:
            return individual + random.uniform(-1.0, 1.0)
        elif random.random() < 0.4:
            return individual + random.uniform(-1.0, 1.0)
        else:
            return individual

    def evaluate_fitness(self, individual, func):
        return func(individual)

    def __str__(self):
        return "BBOOpt: Adaptive Population-Based Optimization using Genetic Algorithm with Black Box Optimization"

    def fitness(self, individual, func):
        return self.evaluate_fitness(individual, func)

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
        if random.random() < 0.5:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.2:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.4:
            x = random.uniform(bounds[0], bounds[1])
    return x

# Description: Adaptive Population-Based Optimization using Genetic Algorithm with Black Box Optimization
# Code: 