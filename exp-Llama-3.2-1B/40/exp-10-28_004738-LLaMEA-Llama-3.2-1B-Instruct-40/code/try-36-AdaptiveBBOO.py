import random
import math

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()
        self.population_size = 50
        self.mutation_rate = 0.01
        self.selection_rate = 0.2
        self.population = [self.generate_individual() for _ in range(self.population_size)]

    def generate_individual(self):
        return [random.uniform(-5.0, 5.0) for _ in range(self.dim)]

    def evaluate_fitness(self, individual, func):
        return func(individual)

    def __call__(self, func):
        return self.evaluate_fitness(self.population[0], func)

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
        if random.random() < 0.5:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.2:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.4:
            x = random.uniform(bounds[0], bounds[1])
    return x

# Description: Adaptive Black Box Optimization using Genetic Algorithm with Mutation and Selection
# Code: 