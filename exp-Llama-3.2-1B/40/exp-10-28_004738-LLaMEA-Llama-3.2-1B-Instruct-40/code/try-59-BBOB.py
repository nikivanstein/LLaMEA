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

def bbo_opt_adapt(func, x0, bounds, budget, strategy):
    x = x0
    best_func = func(x)
    best_x = x
    best_fitness = best_func(x)
    for _ in range(budget):
        if strategy == 'uniform':
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
        elif strategy == 'bounded':
            x = func(x)
            if x < bounds[0]:
                x = bounds[0]
            elif x > bounds[1]:
                x = bounds[1]
            if random.random() < 0.4:
                x = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.2:
                x = random.uniform(bounds[0], bounds[1])
        elif strategy == 'crossover':
            parent1 = func(x0)
            parent2 = func(x)
            child = (parent1 + parent2) / 2
            if random.random() < 0.4:
                child = func(child)
            if child < bounds[0]:
                child = bounds[0]
            elif child > bounds[1]:
                child = bounds[1]
            if random.random() < 0.2:
                child = func(child)
            if child < bounds[0]:
                child = bounds[0]
            elif child > bounds[1]:
                child = bounds[1]
            x = child
        elif strategy =='mutation':
            x = func(x)
            if random.random() < 0.4:
                x = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.2:
                x = random.uniform(bounds[0], bounds[1])
        x = func(x)
        if x < bounds[0]:
            x = bounds[0]
        elif x > bounds[1]:
            x = bounds[1]
        if random.random() < 0.2:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.2:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.2:
            x = random.uniform(bounds[0], bounds[1])
    return x, best_func, best_fitness

# Example usage:
budget = 1000
dim = 5
bounds = (-5.0, 5.0)
x0 = [0.0]
best_individual, best_func, best_fitness = bbo_opt_adapt(f, x0, bounds, budget, strategy='uniform')
print('Optimized function:', best_func)
print('Optimized individual:', best_individual)
print('Optimized fitness:', best_fitness)