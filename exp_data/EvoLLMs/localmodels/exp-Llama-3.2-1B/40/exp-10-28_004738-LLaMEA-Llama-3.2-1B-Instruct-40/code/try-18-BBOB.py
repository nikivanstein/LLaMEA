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

def bbo_opt(func, x0, bounds, budget, mutation_prob=0.4):
    # Initialize the population
    population = [x0] * 100

    for _ in range(budget):
        # Select the best individual
        best_individual = population[np.argmax([self.f(individual) for individual in population])]

        # Generate a new individual with a small probability
        new_individual = best_individual
        if random.random() < mutation_prob:
            # Randomly change a value in the new individual
            idx = random.randint(0, self.dim - 1)
            new_individual[idx] = random.uniform(-5.0, 5.0)

        # Evaluate the new individual
        fitness = self.f(new_individual)

        # Replace the worst individual with the new one
        population[np.argmin([self.f(individual) for individual in population])] = new_individual

        # Check for convergence
        if np.all(population == best_individual):
            break

    return best_individual

# Description: Black Box Optimization using BBOB
# Code: 