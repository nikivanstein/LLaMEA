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

def bbo_opt(func, x0, bounds, budget, adaptation_rate):
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
        if random.random() < adaptation_rate:
            adaptation = np.random.uniform(-0.1, 0.1)
            x = func(x + adaptation)
    return x

# One-line description: Evolutionary Algorithm for Multi-Dimensional Optimization with Adaptation
# Code: 
# ```python
# ```python
# ```python
def adapt_bbo_opt(func, x0, bounds, budget, adaptation_rate):
    return bbo_opt(func, x0, bounds, budget, adaptation_rate)

# Updated code:
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

def adapt_bbo_opt(func, x0, bounds, budget, adaptation_rate):
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
        if random.random() < adaptation_rate:
            adaptation = np.random.uniform(-0.1, 0.1)
            x = func(x + adaptation)
    return x

# One-line description: Evolutionary Algorithm for Multi-Dimensional Optimization with Adaptation
# Code: 
# ```python
# ```python
# ```python
bbo = BBOB(100, 5)
bbo_func = f
bbo.optimization = adapt_bbo_opt
bbo.budget = 100
bbo.dim = 5
bbo.eval_bbo()