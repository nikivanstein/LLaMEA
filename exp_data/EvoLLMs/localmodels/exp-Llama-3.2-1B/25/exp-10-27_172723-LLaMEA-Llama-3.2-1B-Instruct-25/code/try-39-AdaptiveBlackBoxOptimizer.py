import random
import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.search_strategy = 'random'

    def __call__(self, func):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        func_evaluations = self.func_evaluations
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        if self.search_strategy == 'random':
            point = np.random.choice(self.search_space)
        elif self.search_strategy == 'bounded':
            point = np.random.uniform(self.search_space[0], self.search_space[1])
        else:
            raise ValueError("Invalid search strategy")

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

    def adapt_search_strategy(self, fitness):
        if fitness > self.best_fitness:
            self.search_strategy = 'bounded'
            self.best_individual = self.evaluate_fitness(self.best_individual)
            self.best_fitness = fitness
        elif fitness < self.best_fitness:
            self.search_strategy = 'random'
            self.best_individual = self.evaluate_fitness(self.best_individual)
            self.best_fitness = fitness

# One-line description: "Adaptive Black Box Optimizer: An adaptive metaheuristic algorithm that dynamically adjusts its search strategy based on the performance of the current solution"