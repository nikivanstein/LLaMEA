import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

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
        return self.search_space[0], self.search_space[1]

    def novel_metaheuristic(self, func, budget, dim):
        # Initialize the population with random points in the search space
        population = [self.search_space] * self.budget
        for i in range(self.budget):
            population[i] = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
        
        # Evaluate the function for each point in the population
        for _ in range(self.budget):
            # Initialize the minimum and maximum function values
            min_func_value = np.inf
            max_func_value = -np.inf
            # Initialize the best point and its function value
            best_point = None
            best_func_value = -np.inf
            
            # Iterate over the population to find the best point
            for point in population:
                func_value = func(point)
                # Update the minimum and maximum function values
                if func_value < min_func_value:
                    min_func_value = func_value
                    max_func_value = np.inf
                    best_point = point
                elif func_value > max_func_value:
                    max_func_value = func_value
                    best_point = point
            
            # If a better point is found, replace the current point in the population
            if best_point is not None:
                population[_] = best_point
                # Update the function evaluations
                self.func_evaluations += 1
        
        # Return the best point found
        return population[0]