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

    def __call__(self, func, x0, bounds, population_size):
        return func(random.uniform(-5.0, 5.0))

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, population_size, budget):
    population = np.random.uniform(bounds[0], bounds[1], size=(population_size, self.dim))
    for _ in range(budget):
        fitness_values = self.evaluate_fitness(population)
        new_population = np.array([func(x) for x, f in zip(population, fitness_values)])
        population = new_population
        if random.random() < 0.4:
            population = np.random.uniform(bounds[0], bounds[1], size=(population_size, self.dim))
    return population

def mutation(individual, bounds, mutation_prob):
    if random.random() < mutation_prob:
        index = np.random.randint(0, self.dim)
        individual[index] = random.uniform(bounds[0], bounds[1])
    return individual

# Black Box Optimization using BBOB
# Code: 