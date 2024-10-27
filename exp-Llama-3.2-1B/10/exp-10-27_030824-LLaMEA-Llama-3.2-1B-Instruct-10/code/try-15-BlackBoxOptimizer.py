# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
import math
from collections import deque

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.population = deque(maxlen=self.budget)

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def mutate(self, individual):
        # Randomly swap two elements in the individual
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def crossover(self, parent1, parent2):
        # Select a random crossover point
        crossover_point = random.randint(1, len(parent1) - 1)
        # Create a child individual by combining the two parents
        child = list(parent1[:crossover_point])
        for i in range(crossover_point, len(parent2)):
            child.append(parent2[i])
        return child

    def selection(self, population):
        # Select the fittest individuals using tournament selection
        tournament_size = 3
        tournament_results = []
        for _ in range(self.budget):
            individual = random.choice(population)
            for _ in range(tournament_size):
                tournament_results.append(individual)
            winner = tournament_results.index(max(tournament_results))
            population[winner] = individual
        return population

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Code: 
# ```python
import numpy as np

def linear_interpolation(x, y):
    return y * (1 - x) + x

def random_walk(x, y):
    return x + np.random.normal(0, 1) * (y - x)

def bbo_optimization(budget, dim, func, search_space):
    optimizer = BlackBoxOptimizer(budget, dim)
    population = []
    while len(population) < budget:
        individual = random.choice([optimizer, random, linear_interpolation, random_walk])
        if individual is optimizer:
            population.append(individual())
        else:
            population.append(individual())
    return population

# Example usage:
func = lambda x: x**2
search_space = [-5.0, 5.0]
population = bbo_optimization(100, 10, func, search_space)
best_individual = max(population, key=lambda individual: func(individual))
print("Best individual:", best_individual)
print("Best fitness:", func(best_individual))