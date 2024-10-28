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
        if random.random() < 0.4:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.2:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.4:
            x = random.uniform(bounds[0], bounds[1])
    return x

def bbo_optimize(func, x0, bounds, budget):
    # Use a genetic algorithm with mutation and selection
    population = [x0] * 100  # Initial population of 100 individuals
    for _ in range(budget):
        # Select the fittest individuals
        fitness = [func(individual) for individual in population]
        fittest_indices = np.argsort(fitness)[-10:]  # Select the top 10 fittest individuals
        fittest_individuals = [population[i] for i in fittest_indices]
        # Perform mutation and selection
        for _ in range(10):  # Perform 10 generations
            # Select the fittest individuals
            fitness = [func(individual) for individual in fittest_individuals]
            fittest_indices = np.argsort(fitness)[-10:]  # Select the top 10 fittest individuals
            fittest_individuals = [fittest_individuals[i] for i in fittest_indices]
            # Perform crossover and mutation
            for i in range(len(fittest_individuals)):
                parent1, parent2 = fittest_individuals[i], fittest_individuals[(i+1) % len(fittest_individuals)]
                child = (parent1 + parent2) / 2
                if random.random() < 0.2:
                    child = random.uniform(bounds[0], bounds[1])
                if random.random() < 0.4:
                    child = random.uniform(bounds[0], bounds[1])
                population.append(child)
        # Replace the least fit individuals with the new ones
        population = population[:100]
    # Return the fittest individual
    return population[0]

# Example usage
bbo = BBOB(10, 10)
x0 = np.array([1.0, 1.0])
bounds = np.array([[-5.0, -5.0], [5.0, 5.0]])
result = bbo_optimize(f, x0, bounds, 10)
print("Optimal solution:", result)