import numpy as np
from scipy.optimize import minimize
import random
import copy

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function
        self.population = []

    def __call__(self, func, initial_guess, iterations):
        for _ in range(iterations):
            if _ >= self.budget:
                break
            new_population = []
            for _ in range(self.dim):
                new_individual = copy.deepcopy(initial_guess)
                new_individual = self.evaluate_fitness(new_individual)
                new_population.append(new_individual)
            new_population = np.array(new_population)
            new_population = self.budget * new_population
            new_individuals = np.random.choice(new_population, self.dim, replace=True)
            new_population = new_individuals
            self.population.append(new_population)

    def evaluate_fitness(self, individual):
        return self.func(individual)

    def select(self, population):
        selected_individuals = np.random.choice(population, self.budget, replace=False)
        return selected_individuals

    def mutate(self, individual):
        mutated_individual = copy.deepcopy(individual)
        mutated_individual[0] += random.uniform(-0.01, 0.01)
        mutated_individual[1] += random.uniform(-0.01, 0.01)
        return mutated_individual

    def __str__(self):
        return "BlackBoxOptimizer: Novel Metaheuristic Algorithm for Black Box Optimization"

    def __repr__(self):
        return str(self)

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 
# Novel Metaheuristic Algorithm for Black Box Optimization
# 
# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 