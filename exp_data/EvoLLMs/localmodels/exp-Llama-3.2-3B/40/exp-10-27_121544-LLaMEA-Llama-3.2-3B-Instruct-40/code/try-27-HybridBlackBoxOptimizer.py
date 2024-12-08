import numpy as np
from scipy.optimize import differential_evolution
import random

class HybridBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        # Select a random individual from the current population
        individual = np.random.choice(self.x0)

        # Evaluate the fitness of the individual
        fitness = func(individual)

        # If the budget is not exhausted, refine the individual's strategy
        if self.budget > 1:
            # Randomly select two lines of the individual
            line1, line2 = random.sample(range(self.dim), 2)

            # Randomly select a new strategy
            new_strategy = random.choice(['sbx', 'rand1', 'log-uniform'])

            # Create a new individual with the refined strategy
            new_individual = np.copy(individual)
            new_individual[line1] = new_strategy
            new_individual[line2] = new_strategy

            # Evaluate the fitness of the new individual
            new_fitness = func(new_individual)

            # If the new individual's fitness is better, update the individual
            if new_fitness < fitness:
                individual = new_individual

        # Return the individual and its fitness
        return individual, fitness

# Example usage:
def f1(x):
    return np.sum(x**2)

def f2(x):
    return np.sum(x**2) + 2 * np.sum(x)

def f3(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2)

def f4(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x)

def f5(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2)

def f6(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x)

def f7(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x) + 7 * np.sum(x**2)

def f8(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x) + 7 * np.sum(x**2) + 8 * np.sum(x**2)

def f9(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x) + 7 * np.sum(x**2) + 8 * np.sum(x**2) + 9 * np.sum(x**2)

def f10(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x) + 7 * np.sum(x**2) + 8 * np.sum(x**2) + 9 * np.sum(x**2) + 10 * np.sum(x**2)

def f11(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x) + 7 * np.sum(x**2) + 8 * np.sum(x**2) + 9 * np.sum(x**2) + 10 * np.sum(x**2) + 11 * np.sum(x**2)

def f12(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x) + 7 * np.sum(x**2) + 8 * np.sum(x**2) + 9 * np.sum(x**2) + 10 * np.sum(x**2) + 11 * np.sum(x**2) + 12 * np.sum(x**2)

def f13(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x) + 7 * np.sum(x**2) + 8 * np.sum(x**2) + 9 * np.sum(x**2) + 10 * np.sum(x**2) + 11 * np.sum(x**2) + 12 * np.sum(x**2) + 13 * np.sum(x**2)

def f14(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x) + 7 * np.sum(x**2) + 8 * np.sum(x**2) + 9 * np.sum(x**2) + 10 * np.sum(x**2) + 11 * np.sum(x**2) + 12 * np.sum(x**2) + 13 * np.sum(x**2) + 14 * np.sum(x**2)

def f15(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x) + 7 * np.sum(x**2) + 8 * np.sum(x**2) + 9 * np.sum(x**2) + 10 * np.sum(x**2) + 11 * np.sum(x**2) + 12 * np.sum(x**2) + 13 * np.sum(x**2) + 14 * np.sum(x**2) + 15 * np.sum(x**2)

def f16(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x) + 7 * np.sum(x**2) + 8 * np.sum(x**2) + 9 * np.sum(x**2) + 10 * np.sum(x**2) + 11 * np.sum(x**2) + 12 * np.sum(x**2) + 13 * np.sum(x**2) + 14 * np.sum(x**2) + 15 * np.sum(x**2) + 16 * np.sum(x**2)

def f17(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x) + 7 * np.sum(x**2) + 8 * np.sum(x**2) + 9 * np.sum(x**2) + 10 * np.sum(x**2) + 11 * np.sum(x**2) + 12 * np.sum(x**2) + 13 * np.sum(x**2) + 14 * np.sum(x**2) + 15 * np.sum(x**2) + 16 * np.sum(x**2) + 17 * np.sum(x**2)

def f18(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x) + 7 * np.sum(x**2) + 8 * np.sum(x**2) + 9 * np.sum(x**2) + 10 * np.sum(x**2) + 11 * np.sum(x**2) + 12 * np.sum(x**2) + 13 * np.sum(x**2) + 14 * np.sum(x**2) + 15 * np.sum(x**2) + 16 * np.sum(x**2) + 17 * np.sum(x**2) + 18 * np.sum(x**2)

def f19(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x) + 7 * np.sum(x**2) + 8 * np.sum(x**2) + 9 * np.sum(x**2) + 10 * np.sum(x**2) + 11 * np.sum(x**2) + 12 * np.sum(x**2) + 13 * np.sum(x**2) + 14 * np.sum(x**2) + 15 * np.sum(x**2) + 16 * np.sum(x**2) + 17 * np.sum(x**2) + 18 * np.sum(x**2) + 19 * np.sum(x**2)

def f20(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x) + 7 * np.sum(x**2) + 8 * np.sum(x**2) + 9 * np.sum(x**2) + 10 * np.sum(x**2) + 11 * np.sum(x**2) + 12 * np.sum(x**2) + 13 * np.sum(x**2) + 14 * np.sum(x**2) + 15 * np.sum(x**2) + 16 * np.sum(x**2) + 17 * np.sum(x**2) + 18 * np.sum(x**2) + 19 * np.sum(x**2) + 20 * np.sum(x**2)

def f21(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x) + 7 * np.sum(x**2) + 8 * np.sum(x**2) + 9 * np.sum(x**2) + 10 * np.sum(x**2) + 11 * np.sum(x**2) + 12 * np.sum(x**2) + 13 * np.sum(x**2) + 14 * np.sum(x**2) + 15 * np.sum(x**2) + 16 * np.sum(x**2) + 17 * np.sum(x**2) + 18 * np.sum(x**2) + 19 * np.sum(x**2) + 20 * np.sum(x**2) + 21 * np.sum(x**2)

def f22(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x) + 7 * np.sum(x**2) + 8 * np.sum(x**2) + 9 * np.sum(x**2) + 10 * np.sum(x**2) + 11 * np.sum(x**2) + 12 * np.sum(x**2) + 13 * np.sum(x**2) + 14 * np.sum(x**2) + 15 * np.sum(x**2) + 16 * np.sum(x**2) + 17 * np.sum(x**2) + 18 * np.sum(x**2) + 19 * np.sum(x**2) + 20 * np.sum(x**2) + 21 * np.sum(x**2) + 22 * np.sum(x**2)

def f23(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x) + 7 * np.sum(x**2) + 8 * np.sum(x**2) + 9 * np.sum(x**2) + 10 * np.sum(x**2) + 11 * np.sum(x**2) + 12 * np.sum(x**2) + 13 * np.sum(x**2) + 14 * np.sum(x**2) + 15 * np.sum(x**2) + 16 * np.sum(x**2) + 17 * np.sum(x**2) + 18 * np.sum(x**2) + 19 * np.sum(x**2) + 20 * np.sum(x**2) + 21 * np.sum(x**2) + 22 * np.sum(x**2) + 23 * np.sum(x**2)

def f24(x):
    return np.sum(x**2) + 2 * np.sum(x) + 3 * np.sum(x**2) + 4 * np.sum(x) + 5 * np.sum(x**2) + 6 * np.sum(x) + 7 * np.sum(x**2) + 8 * np.sum(x**2) + 9 * np.sum(x**2) + 10 * np.sum(x**2) + 11 * np.sum(x**2) + 12 * np.sum(x**2) + 13 * np.sum(x**2) + 14 * np.sum(x**2) + 15 * np.sum(x**2) + 16 * np.sum(x**2) + 17 * np.sum(x**2) + 18 * np.sum(x**2) + 19 * np.sum(x**2) + 20 * np.sum(x**2) + 21 * np.sum(x**2) + 22 * np.sum(x**2) + 23 * np.sum(x**2) + 24 * np.sum(x**2)

optimizer = HybridBlackBoxOptimizer(budget=10, dim=10)
for i in range(10):
    func_name = f"f{i+1}"
    func = globals()[func_name]
    individual, fitness = optimizer(func)
    print(f"Individual: {individual}, Fitness: {fitness}")