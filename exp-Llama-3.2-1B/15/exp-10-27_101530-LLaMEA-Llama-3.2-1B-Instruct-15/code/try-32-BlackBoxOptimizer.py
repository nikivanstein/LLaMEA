import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.budget_ratio = 0.15
        self.best_individual = None

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
        # If the budget is reached, return the best point found so far
        if self.best_individual is None:
            self.best_individual = self.search_space[0], self.search_space[1]
        return self.best_individual

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.optimizer = BlackBoxOptimizer(budget, dim)
        self.population_size = 100
        self.population = np.random.uniform(self.search_space[0], self.search_space[1], (self.population_size, self.dim))
        self.mutation_rate = 0.01
        self.selection_rate = 0.5

    def __call__(self, func):
        while True:
            # Evaluate the function for the current population
            aucs = np.array([self.optimizer(func(individual)) for individual in self.population])
            # Select the best individual
            best_individual = self.population[np.argmax(aucs)]
            # Select the next generation
            self.population = np.random.choice(self.population_size, self.population_size, p=[1 - self.selection_rate, self.selection_rate])
            # Mutate the selected individuals
            self.population = np.concatenate((self.population, [best_individual] * (self.population_size - self.population_size / 2)))
            # Evaluate the function for the new generation
            aucs = np.array([self.optimizer(func(individual)) for individual in self.population])
            # Replace the old population with the new one
            self.population = self.population[:self.population_size]
            # Update the best individual
            self.best_individual = best_individual
            # Check if the budget is reached
            if self.func_evaluations >= self.budget:
                return self.best_individual

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# Optimizes black box functions using a novel metaheuristic algorithm
# with a probability of 0.15 to refine its strategy