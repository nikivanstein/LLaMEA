import random
import numpy as np

class DERE:
    def __init__(self, budget, dim, max_iter=1000):
        self.budget = budget
        self.dim = dim
        self.max_iter = max_iter
        self.funcs = self.generate_functions()
        self.funcs = {func.__name__: func for func in self.funcs}

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func, x0, bounds, iteration=0):
        if iteration >= self.max_iter:
            return x0

        x = x0
        for _ in range(self.budget):
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

    def mutate(self, func, x, mutation_prob=0.2):
        x_prime = x + random.uniform(-mutation_prob, mutation_prob)
        if random.random() < 0.5:
            x_prime = random.uniform(bounds[0], bounds[1])
        return x_prime

    def refine(self, func, x, bounds, mutation_prob=0.2):
        x_prime = self.mutate(func, x, mutation_prob)
        if random.random() < 0.4:
            x_prime = self.mutate(func, x, mutation_prob)
        if random.random() < 0.4:
            x_prime = self.mutate(func, x, mutation_prob)
        return x_prime

    def evaluate_fitness(self, func, x, bounds):
        return func(x)

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget, mutation_prob=0.2, iteration=0):
    x = x0
    for _ in range(budget):
        x = func(x, x0, bounds, iteration)
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
        new_individual = DERE(budget, dim, iteration+1)(func, x, bounds, mutation_prob)
        x = new_individual
    return x

# Initialize the Black Box Optimization using BBOB algorithm
bbo = BBOB(budget=1000, dim=5)

# Evaluate the function f(x) = x^2 + 0.5x + 0.1
f_x = f(0)
f_x_prime = f_prime(f_x)
f_double_prime_x = f_double_prime(f_x)
f_double_prime_prime_x = f_double_prime_prime(f_x)

# Optimize the function f(x) = x^2 + 0.5x + 0.1 using DBOB
x_opt = bbo(BOBO)(f_x, f_x_prime, f_double_prime_x, f_double_prime_prime_x, mutation_prob=0.2)

# Print the optimized solution
print("Optimized Solution:", x_opt)

# Evaluate the function f(x) = x^2 + 0.5x + 0.1
f_x_opt = f(x_opt)
f_x_prime_opt = f_prime(f_x_opt)
f_double_prime_x_opt = f_double_prime(f_x_opt)
f_double_prime_prime_x_opt = f_double_prime_prime(f_x_opt)

# Print the optimized solution
print("Optimized Solution:", x_opt)