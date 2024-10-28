import random
import numpy as np

class MultiOptimization:
    def __init__(self, budget, dim, num_solutions):
        self.budget = budget
        self.dim = dim
        self.num_solutions = num_solutions
        self.solutions = self.generate_solutions()

    def generate_solutions(self):
        solutions = []
        for _ in range(self.num_solutions):
            solution = []
            for _ in range(self.dim):
                solution.append(random.uniform(-5.0, 5.0))
            solutions.append(solution)
        return solutions

    def __call__(self, func):
        for solution in self.solutions:
            new_individual = func(solution)
            if new_individual < self.budget:
                self.solutions.append(new_individual)
        return self.solutions

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
        if random.random() < 0.4:
            x = random.uniform(bounds[0], bounds[1])
    return x

# Description: Evolutionary Algorithm for Multi-Optimization Problem
# Code: 