# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np

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

    def __call__(self, func, mutation_rate=0.1, crossover_rate=0.1):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.
            mutation_rate (float, optional): The rate at which to introduce mutation. Defaults to 0.1.
            crossover_rate (float, optional): The rate at which to perform crossover. Defaults to 0.1.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

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

        # Perform mutation
        if random.random() < mutation_rate:
            # Randomly select a point in the search space
            point = self.search_space[np.random.randint(0, self.dim)]

            # Evaluate the function at the new point
            new_value = func(point)

            # If the new value is better than the best value found so far,
            # update the best value and its corresponding index
            if new_value > best_value:
                best_value = new_value
                best_index = point

        # Perform crossover
        if random.random() < crossover_rate:
            # Select two random points in the search space
            point1 = self.search_space[np.random.randint(0, self.dim)]
            point2 = self.search_space[np.random.randint(0, self.dim)]

            # Evaluate the function at the two points
            value1 = func(point1)
            value2 = func(point2)

            # If the two values are different, perform crossover
            if value1!= value2:
                # Perform crossover
                child1 = (point1 + point2) / 2
                child2 = 2 * point1 - point2

                # Evaluate the function at the child points
                value1 = func(child1)
                value2 = func(child2)

                # If the two values are different, choose the better one
                if value1 > value2:
                    best_value = value1
                    best_index = point1
                else:
                    best_value = value2
                    best_index = point2

        # Return the optimized value
        return best_value

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# 
# This algorithm uses a combination of mutation and crossover to optimize black box functions.
# 
# Parameters:
#     func (callable): The black box function to optimize.
#     mutation_rate (float): The rate at which to introduce mutation. Defaults to 0.1.
#     crossover_rate (float): The rate at which to perform crossover. Defaults to 0.1.
# 
# Returns:
#     float: The optimized value of the function.