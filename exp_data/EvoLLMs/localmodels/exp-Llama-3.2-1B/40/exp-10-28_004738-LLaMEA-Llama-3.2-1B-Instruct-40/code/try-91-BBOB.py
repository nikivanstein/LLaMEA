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
        if random.random() < 0.2:
            x = random.uniform(bounds[0], bounds[1])
    return x

# One-line description: Evolutionary Algorithm for Multi-Dimensional Optimization
# Code: 
# ```python
class EvolutionaryOptimization:
    def __init__(self, budget, dim, alpha=0.5, beta=0.2, mu=0.1, sigma=0.01, max_iter=1000, c1=0.9, c2=2, n_pop=100, n_mates=10):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.sigma = sigma
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.n_pop = n_pop
        self.n_mates = n_mates

    def __call__(self, func, x0, bounds):
        population = np.random.choice([func(x) for x in x0], size=self.n_pop, replace=False)
        fitnesses = [population[i] for i in range(self.n_pop)]
        while len(fitnesses) < self.budget:
            for _ in range(self.max_iter):
                if random.random() < self.alpha:
                    x0 = np.random.uniform(bounds[0], bounds[1], self.dim)
                else:
                    x0 = population[np.argmax(fitnesses)]
                fitnesses.append(func(x0))
            population = np.random.choice([func(x) for x in x0], size=self.n_pop, replace=False)
            fitnesses = [population[i] for i in range(self.n_pop)]
        return population[np.argmax(fitnesses)]

# One-line description: Evolutionary Algorithm for Multi-Dimensional Optimization
# Code: 
# ```python
def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget):
    return EvolutionaryOptimization(budget, dim, alpha=0.5, beta=0.2, mu=0.1, sigma=0.01, max_iter=1000, c1=0.9, c2=2, n_pop=100, n_mates=10)(func, x0, bounds)

# Example usage:
bbo = bbo_opt(f, np.array([-5.0, -5.0]), np.array([-5.0, 5.0]), 100)
print(bbo(np.array([-5.0, -5.0])))