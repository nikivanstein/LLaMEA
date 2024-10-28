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
    population_size = 100
    population = [x0] * population_size
    for _ in range(budget):
        fitness_values = [func(individual) for individual in population]
        fitness_values.sort(key=lambda x: x, reverse=True)
        selected_individuals = fitness_values[:population_size // 2]
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = random.sample(selected_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < 0.4:
                child = parent1
            if random.random() < 0.2:
                child = parent2
            if random.random() < 0.4:
                child = random.uniform(bounds[0], bounds[1])
            new_population.append(child)
        population = new_population
    return population[0]

# BBOB Algorithm Code
def bbo_optimize(bbo, func, x0, budget):
    best_individual = bbo_func(func, x0, bbo.dim)
    best_fitness = bbo_func(best_individual, x0, bbo.dim)
    return best_individual, best_fitness

def bbo_func(func, x, dim):
    return func(x)

# Initialize the BBOB problem
bbo = BBO()
x0 = np.array([0.5])
budget = 100
best_individual, best_fitness = bbo_optimize(bbo, f, x0, budget)

# Update the BBOB problem
x0 = np.array([best_individual])
budget = 100
best_individual, best_fitness = bbo_optimize(bbo, f, x0, budget)

# Print the results
print(f"Best Individual: {best_individual}, Best Fitness: {best_fitness}")