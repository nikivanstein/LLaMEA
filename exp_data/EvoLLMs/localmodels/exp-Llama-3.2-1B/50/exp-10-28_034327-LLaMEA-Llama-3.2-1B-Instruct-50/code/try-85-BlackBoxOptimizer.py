import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.cooling_rate = 0.01

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

    def select_next_individual(self):
        if self.best_individual is None:
            return self.population
        else:
            # Select the next individual using the Iterated Permutation and Cooling Algorithm
            next_individual = np.random.choice(self.population, p=[0.6, 0.4])
            return next_individual

    def mutate(self, individual):
        # Randomly swap two elements in the individual
        if random.random() < self.cooling_rate:
            i, j = random.sample(range(self.dim), 2)
            self.population[self.population.index(individual)] = individual[i]
            self.population[self.population.index(individual + 1)] = individual[j]
            self.population[self.population.index(individual + 1)] = individual[i]
        return individual

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
def evaluate_bbof(func, bounds, initial_population, budget, cooling_rate):
    optimizer = BlackBoxOptimizer(budget, len(bounds))
    for _ in range(budget):
        best_individual = optimizer.best_individual
        best_fitness = optimizer.best_fitness
        for individual in initial_population:
            fitness = func(individual)
            if fitness > best_fitness:
                best_individual = individual
                best_fitness = fitness
        optimizer.population = [individual for individual in optimizer.population if -5.0 <= individual[0] <= 5.0 and -5.0 <= individual[1] <= 5.0]
        optimizer.best_individual = best_individual
        optimizer.best_fitness = best_fitness
        if best_fitness == np.max(func(np.random.uniform(-5.0, 5.0, len(bounds)))):
            return best_individual
    return None