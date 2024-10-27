# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from copy import deepcopy
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

        # Initialize a queue for the genetic algorithm
        queue = deque([(func, 0, best_value)])

        # Initialize the population
        population = [func(np.array([random.uniform(-5.0, 5.0)]) for _ in range(self.dim)) for _ in range(100)]

        while queue and len(population) < self.budget:
            # Dequeue the individual with the highest fitness value
            individual, fitness, _ = queue.popleft()

            # Evaluate the function at the current point
            point = self.search_space[np.random.randint(0, self.dim)]

            # Evaluate the function at the current point
            value = func(point)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > fitness:
                best_value = value
                best_index = point
                population = [func(np.array([random.uniform(-5.0, 5.0)]) for _ in range(self.dim)) for _ in range(100)]
            else:
                population[population.index(func(point))] = func(point)

        # Return the optimized value
        return best_value

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# This algorithm uses a genetic algorithm to optimize a black box function by evolving a population of individuals with better fitness values