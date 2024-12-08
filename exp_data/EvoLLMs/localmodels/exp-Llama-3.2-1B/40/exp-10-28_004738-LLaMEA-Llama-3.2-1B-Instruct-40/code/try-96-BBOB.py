import random
import numpy as np

class BBOB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()
        self.population = []

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

def bbo_optimize(func, x0, bounds, budget, dim):
    # Select the initial population using tournament selection
    tournament_size = 5
    tournament_population = []
    for _ in range(budget):
        tournament_population.append(bbo_opt(func, x0, bounds, 1))
        x0 = tournament_population[_]
    # Select the fittest individuals
    population = tournament_population
    while len(population) < budget:
        individual = random.choice(population)
        tournament_population.append(bbo_opt(func, individual, bounds, 1))
        population.remove(individual)
    # Select the fittest individuals
    population = tournament_population
    while len(population) < budget:
        individual = random.choice(population)
        tournament_population.append(bbo_opt(func, individual, bounds, 1))
        population.remove(individual)
    # Select the fittest individuals
    population = tournament_population
    while len(population) < budget:
        individual = random.choice(population)
        tournament_population.append(bbo_opt(func, individual, bounds, 1))
        population.remove(individual)
    # Perform evolution
    while len(population) < budget:
        x0 = random.uniform(bounds[0], bounds[1])
        for individual in population:
            if random.random() < 0.4:
                x0 = individual
            else:
                x0 = bbo_opt(func, x0, bounds, 1)
        population.append(x0)
    # Evaluate the objective function on the final population
    score = 0
    for individual in population:
        score += bbo_opt(func, individual, bounds, 1)
    return score

# One-line description: Evolutionary Algorithm for Multi-Dimensional Optimization
# Code: 