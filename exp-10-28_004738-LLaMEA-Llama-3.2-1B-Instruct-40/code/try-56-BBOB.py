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
    # Novel heuristic algorithm: Genetic Algorithm with Evolutionary Strategy
    # Description: Black Box Optimization using BBOB
    # Code: 
    population = [x0] * budget
    for _ in range(budget):
        fitnesses = [func(individual) for individual in population]
        selection_probabilities = np.array([fitness / sum(fitnesses) for fitness in fitnesses])
        selection_probabilities = selection_probabilities[:, np.newaxis]
        selection_indices = np.random.choice(len(population), size=budget, replace=False, p=selection_probabilities)
        population = [population[i] for i in selection_indices]
        new_population = []
        for _ in range(100):  # Evolve population for 100 generations
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = parent1[:len(parent1)//2] + parent2[len(parent1)//2:]
            child = func(child)
            if random.random() < 0.4:
                child = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.2:
                child = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.4:
                child = random.uniform(bounds[0], bounds[1])
            new_population.append(child)
        population = new_population
    return population[np.argmax(fitnesses)]

# Test the algorithm
bbo = BBOB(100, 10)
best_func = bbo_opt(f, np.array([-5.0, -5.0]), [[-5.0, -5.0], [5.0, 5.0]], 100)
print("Best function:", best_func)
print("Best fitness:", best_func(best_func))