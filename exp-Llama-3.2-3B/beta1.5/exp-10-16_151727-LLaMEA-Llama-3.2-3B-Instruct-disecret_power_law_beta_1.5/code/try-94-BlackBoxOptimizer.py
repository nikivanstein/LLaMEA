import numpy as np
from scipy.optimize import differential_evolution
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(-5.0, 5.0)] * self.dim
        res = differential_evolution(func, bounds, x0=np.array([np.random.uniform(-5.0, 5.0) for _ in range(self.dim)]), maxiter=self.budget)
        return res.x

def adaptive_crossover(x1, x2):
    if np.random.rand() < 0.5:
        return (x1 + x2) / 2
    else:
        return x1

def adaptive_mutation(x, func, bounds):
    if np.random.rand() < 0.1:
        x += np.random.uniform(-1, 1, size=self.dim)
        x = np.clip(x, bounds[0], bounds[1])
    return x

def adaptive_optimize(func, bounds, dim, budget):
    optimizer = BlackBoxOptimizer(budget, dim)
    x = np.array([np.random.uniform(bounds[0][i], bounds[1][i]) for i in range(dim)])
    population = [x]
    for _ in range(budget):
        new_individual = np.array([np.random.uniform(bounds[0][i], bounds[1][i]) for i in range(dim)])
        for individual in population:
            new_individual = adaptive_crossover(new_individual, individual)
            new_individual = adaptive_mutation(new_individual, func, bounds)
            if func(new_individual) < func(individual):
                population.remove(individual)
                population.append(new_individual)
        population.sort(key=lambda x: func(x))
        if len(population) > budget:
            population = population[:budget]
        x = population[0]
    return x

def probabilistic_selection(population):
    selection_probabilities = [func(individual) for individual in population]
    selection_probabilities = np.array(selection_probabilities) / np.sum(selection_probabilities)
    selected_index = np.random.choice(len(population), size=1, p=selection_probabilities)
    return population[selected_index]

def adaptive_optimize_probabilistic(func, bounds, dim, budget):
    optimizer = BlackBoxOptimizer(budget, dim)
    x = np.array([np.random.uniform(bounds[0][i], bounds[1][i]) for i in range(dim)])
    population = [x]
    for _ in range(budget):
        new_individual = np.array([np.random.uniform(bounds[0][i], bounds[1][i]) for i in range(dim)])
        for individual in population:
            new_individual = adaptive_crossover(new_individual, individual)
            new_individual = adaptive_mutation(new_individual, func, bounds)
            if func(new_individual) < func(individual):
                population.remove(individual)
                population.append(new_individual)
        population.sort(key=lambda x: func(x))
        if len(population) > budget:
            population = population[:budget]
        selected_individual = probabilistic_selection(population)
        x = selected_individual
    return x

# Example usage
def func(x):
    return np.sum(x**2)

bounds = [(-5.0, 5.0)] * 10
dim = 10
budget = 100

x = adaptive_optimize_probabilistic(func, bounds, dim, budget)
print(x)