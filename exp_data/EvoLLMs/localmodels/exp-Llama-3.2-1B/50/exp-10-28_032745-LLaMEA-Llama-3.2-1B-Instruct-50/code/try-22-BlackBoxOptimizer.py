import numpy as np
from scipy.optimize import minimize
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function

    def __call__(self, func, initial_guess, iterations):
        population_size = 100
        for _ in range(iterations):
            population = [initial_guess]
            for _ in range(population_size):
                fitness = func(population[-1], random.uniform(self.search_space[0], self.search_space[1]))
                new_individual = population[-1] + [x + random.uniform(-0.01, 0.01) for x in population[-1]]
                population.append(new_individual)
                fitness = func(new_individual, random.uniform(self.search_space[0], self.search_space[1]))
            population = population[:population_size]
            population = np.array(population)
            population = population[np.argsort(population[:, 2])]
            population = population[:population_size]
            population = population[np.argsort(population[:, 2])]
            new_individual = population[-1]
            new_fitness = func(new_individual, random.uniform(self.search_space[0], self.search_space[1]))
            if new_fitness < population[-2, 2]:
                population = population[:-1]
                population = population[np.argsort(population[:, 2])]
                population = population[:population_size]
                population = population[np.argsort(population[:, 2])]
                new_individual = population[-1]
                fitness = func(new_individual, random.uniform(self.search_space[0], self.search_space[1]))
                population = np.array([new_individual])
                population = population[np.argsort(population[:, 2])]
                population = population[:population_size]
                population = population[np.argsort(population[:, 2])]
                population[-1] = new_individual
                population[-1, 2] = new_fitness
            else:
                population[-1, 2] = new_fitness
            population = population[:population_size]
            population = np.array(population)
            population = population[np.argsort(population[:, 2])]
            population = population[:population_size]
            population = population[np.argsort(population[:, 2])]
            if population[-1, 2] < self.func(population[-2], random.uniform(self.search_space[0], self.search_space[1])):
                population = population[:-1]
                population = population[np.argsort(population[:, 2])]
                population = population[:population_size]
                population = population[np.argsort(population[:, 2])]
                population[-1, 2] = population[-2, 2]
                population[-1] = population[-1, :]

    def evaluate_fitness(self, individual):
        return self.func(individual, random.uniform(self.search_space[0], self.search_space[1]))

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 