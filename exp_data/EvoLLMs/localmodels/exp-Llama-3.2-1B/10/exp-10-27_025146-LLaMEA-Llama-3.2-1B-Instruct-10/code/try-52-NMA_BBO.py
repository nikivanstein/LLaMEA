import random
import numpy as np

class NMA_BBO:
    def __init__(self, budget, dim):
        """
        Initialize the NMA_BBO with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func):
        """
        Optimize the black box function using the NMA_BBO.

        Args:
            func (callable): The black box function to optimize.

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

        # Use simulated annealing to refine the solution
        if np.random.rand() < 0.1:
            # Generate a new point in the search space using the current best point
            new_point = self.search_space[best_index]

            # Generate a new random point in the search space
            new_point = self.search_space[np.random.randint(0, self.dim)]

            # Evaluate the function at the current and new points
            value1 = func(point)
            value2 = func(new_point)

            # If the new point is better than the current point, update the best point
            if value2 > value1:
                best_index = new_point
        else:
            # Evaluate the function at the current point
            value = func(point)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > best_value:
                best_value = value
                best_index = point

        # Return the optimized value
        return best_value