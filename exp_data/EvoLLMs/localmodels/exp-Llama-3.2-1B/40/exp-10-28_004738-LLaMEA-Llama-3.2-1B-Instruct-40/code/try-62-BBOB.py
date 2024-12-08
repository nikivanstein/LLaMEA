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
    population = [x]
    for _ in range(budget):
        new_individual = self.evaluate_fitness(population[-1])
        if random.random() < 0.4:
            new_individual = self.refine_strategy(func, x, bounds, population[-1], new_individual)
        population.append(new_individual)
    return population[-1]

def evaluate_fitness(individual, bounds, func):
    return func(individual)

def refine_strategy(func, x, bounds, individual, new_individual):
    # Select a new individual based on the probability of mutation
    if random.random() < 0.2:
        new_individual = self.evaluate_fitness(new_individual)
        if new_individual < bounds[0]:
            new_individual = bounds[0]
        elif new_individual > bounds[1]:
            new_individual = bounds[1]
        if random.random() < 0.1:
            new_individual = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.15:
            new_individual = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.2:
            new_individual = random.uniform(bounds[0], bounds[1])
    return new_individual

# Example usage:
bbo = BBOB(100, 5)
problem = RealSingleObjectiveProblem(5, "Sphere", 1.0)
bbo.optimize(problem, bbo_opt, f, f_prime, f_double_prime, f_double_prime_prime, bounds=[-5.0, 5.0], budget=100)