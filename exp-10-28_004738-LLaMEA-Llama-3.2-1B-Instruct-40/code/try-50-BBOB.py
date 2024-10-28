import random
import numpy as np
from scipy.optimize import minimize

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
        if random.random() < 0.4:  # Refine strategy with probability 0.4
            x = np.random.uniform(bounds[0], bounds[1])
        if random.random() < 0.2:  # Randomly swap bounds with probability 0.2
            x = bounds[1], bounds[0]
        if random.random() < 0.4:  # Randomly swap bounds with probability 0.4
            x = bounds[0], bounds[1]
    return x

def bbo_optimize(bbo, func, x0, bounds, budget):
    return bbo(bbo_opt(func, x0, bounds, budget), x0)

# Initialize the problem
problem = BBOB(100, 5)  # 100 function evaluations, 5 dimensions
bbo = BBOB(100, 5)  # Same problem as before

# Evaluate the function 100 times
bbo.evaluate_fitness(lambda x: f(x))

# Optimize the function using the new algorithm
new_individual = bbo_optimize(bbo, f, -4.521232642195706, [-5.0, 5.0], 100)

# Update the selected solution
bbo.funcs = [f(x) for x in new_individual]
bbo.budget = 100

# Print the updated problem
print("Updated problem:")
print("Functions:", bbo.funcs)
print("Budget:", bbo.budget)

# Evaluate the function 100 times again
bbo.evaluate_fitness(lambda x: f(x))

# Optimize the function using the new algorithm again
new_individual = bbo_optimize(bbo, f, -4.521232642195706, [-5.0, 5.0], 100)

# Print the updated problem again
print("Updated problem:")
print("Functions:", bbo.funcs)
print("Budget:", bbo.budget)