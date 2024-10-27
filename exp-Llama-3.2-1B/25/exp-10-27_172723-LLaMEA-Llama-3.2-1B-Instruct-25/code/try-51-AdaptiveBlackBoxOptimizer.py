import random
import numpy as np
from collections import deque

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.search_history = deque(maxlen=self.budget)

    def __call__(self, func):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        func_evaluations = self.func_evaluations
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

    def update_search_strategy(self, new_individual):
        # Calculate the fitness of the new individual
        fitness = self.evaluate_fitness(new_individual)

        # Get the current best individual and its fitness
        best_individual = self.search_history[0]
        best_fitness = fitness

        # If the new individual's fitness is better than the current best individual's fitness
        if fitness > best_fitness:
            # Update the best individual and its fitness
            best_individual = new_individual
            best_fitness = fitness

        # Update the search history with the new individual's fitness
        self.search_history.append((best_individual, fitness))

        # If the number of evaluations has reached the budget
        if len(self.search_history) == self.budget:
            # Refine the search strategy by changing the initial point to the current best individual
            self.search_space = np.linspace(self.best_individual[0], best_individual[0], self.dim)
            self.func_evaluations = 0

        # Update the individual's point to the current best individual
        self.func_evaluations = 0
        self.best_individual = best_individual

# One-line description: "Adaptive Black Box Optimizer: An adaptive metaheuristic algorithm that learns to adapt its search strategy based on the performance of the current solution"