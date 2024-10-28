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
        if random.random() < 0.3:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.2:
            x = random.uniform(bounds[0], bounds[1])
    return x

def mutate(individual, bounds):
    x = individual
    for _ in range(random.randint(1, 5)):
        if random.random() < 0.5:
            x = random.uniform(bounds[0], bounds[1])
        elif random.random() < 0.3:
            x = random.uniform(bounds[0], bounds[1])
    return x

def bbo_optimize(bbo, func, x0, budget):
    population = [bbo.__call__(func) for _ in range(budget)]
    population = np.array(population)
    while True:
        new_population = []
        for _ in range(budget):
            new_individual = mutate(population[_], bbo.bounds)
            new_population.append(new_individual)
        new_population = np.array(new_population)
        new_population = bbo_opt(func, x0, bbo.bounds, budget)
        new_population = np.array(new_population)
        if np.allclose(population, new_population):
            break
        population = new_population
    return population

# Initialize the Black Box Optimization using BBOB algorithm
bbo = BBOB(100, 5)
x0 = np.array([-4.521232642195706, -4.521232642195706, -4.521232642195706, -4.521232642195706, -4.521232642195706])
budget = 100
dim = 5
population = bbo_optimize(bbo, f, x0, budget)
print("Optimized solution:", population)