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
            func = lambda x: np.random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func):
        return func(np.random.uniform(-5.0, 5.0))

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
        if random.random() < 0.1:
            x = random.uniform(bounds[0], bounds[1])
    return x

def bbo_optimize(func, x0, bounds, budget, dim):
    # Novel heuristic algorithm: Evolutionary Black Box Optimization using BBOB
    # Description: Evolutionary Black Box Optimization using BBOB
    # Code:
    population = []
    for _ in range(1000):
        # Select the best individual from the current population
        new_individual = bbo_opt(func, x0, bounds, budget)
        population.append(new_individual)
        
        # Evolve the population using mutation and selection
        if random.random() < 0.1:
            # Mutation: Randomly change a single element in the individual
            index = random.randint(0, dim-1)
            new_individual[index] = np.random.uniform(-5.0, 5.0)
        
        # Selection: Select the fittest individuals for the next generation
        fittest_individuals = sorted(population, key=lambda x: np.mean(np.abs(x - x0)), reverse=True)
        population = fittest_individuals[:int(0.6*budget)]

    # Return the best individual found
    best_individual = population[0]
    return best_individual

# Usage
bbo = BBOB(100, 10)
best_individual = bbo_optimize(f, [-4.521232642195706], [-5.0, 5.0], 100, 10)
print("Best individual:", best_individual)
print("Best fitness:", np.mean(np.abs(best_individual - np.array([-4.521232642195706]))))