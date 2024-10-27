# Description: Novel Metaheuristic Algorithm for Black Box Optimization (NMABBO)
# Code: 
# ```python
import random
import numpy as np
import math
from collections import deque

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Initialize the population of individuals
        population = []

        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Generate a random point in the search space
            point = self.search_space[np.random.randint(0, self.dim)]

            # Evaluate the function at the current point
            value = func(point)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > best_value:
                best_value = value
                best_index = point

            # Create a new individual by changing one element of the current point
            new_individual = point.copy()
            new_individual[0] += random.uniform(-1, 1)  # Change the first element

            # Add the new individual to the population
            population.append(new_individual)

            # If the population exceeds the budget, remove the oldest individual
            if len(population) > self.budget:
                population.pop(0)

        # Return the optimized value
        return best_value

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization