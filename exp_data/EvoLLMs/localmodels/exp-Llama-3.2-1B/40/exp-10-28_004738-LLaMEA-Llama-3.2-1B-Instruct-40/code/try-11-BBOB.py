# Description: Novel Metaheuristic for Black Box Optimization
# Code: 
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
            func = lambda x: np.random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func, x0, bounds, budget):
        new_individual = x0
        for _ in range(budget):
            new_individual = func(new_individual)
            if new_individual < bounds[0]:
                new_individual = bounds[0]
            elif new_individual > bounds[1]:
                new_individual = bounds[1]
            if random.random() < 0.4:
                new_individual = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.6:
                new_individual = random.uniform(bounds[0], bounds[1])
        return new_individual

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget):
    return BBOB(budget, x0).__call__(func, x0, bounds, budget)

# Initial evaluation of BBOB
bbo = BBOB(1000, 5)
print("Initial Evaluation of BBOB:", bbo.budget, "evaluations")

# Selection of a new solution
new_individual = bbo_opt(f, -4.521232642195706, [-5.0, 5.0], 1000)

# Update the population with the new solution
bbo.budget = 1000
bbo.funcs = bbo.funcs + [f]
print("Updated BBOB:", bbo.budget, "evaluations")