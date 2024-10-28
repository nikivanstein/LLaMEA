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
        if random.random() < 0.4:  # Refine the strategy by changing the probability
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.2:  # Refine the strategy by changing the probability
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.1:  # Refine the strategy by changing the probability
            x = random.uniform(bounds[0], bounds[1])
    return x

# Novel Heuristic Algorithm: Evolutionary Algorithm using Evolutionary Strategies
class EvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()
        self.population = [self.generate_individual() for _ in range(100)]  # Initial population

    def generate_individual(self):
        return [random.uniform(-5.0, 5.0) for _ in range(self.dim)]

    def evaluate_fitness(self, individual, func):
        return func(individual)

    def __call__(self, func):
        return self.bbo_opt(func, self.population[0], self.funcs, self.budget)

# One-line description with the main idea
# Evolutionary Algorithm using Evolutionary Strategies
# 
# This algorithm uses an evolutionary strategy to optimize the given black box function.
# The strategy involves generating initial individuals, evaluating their fitness, and then iteratively refining the strategy to improve the fitness.