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

        # Perform simulated annealing to refine the search
        temperature = 1000
        cooling_rate = 0.99
        for _ in range(self.budget):
            # Generate a new point using the current point and a random direction
            new_point = point + np.random.uniform(-5.0, 5.0, self.dim)
            new_value = func(new_point)

            # If the new value is better, accept it with probability 1
            if new_value > best_value:
                best_value = new_value
                best_index = new_point

            # If the new value is worse, accept it with probability 1
            elif np.random.rand() < np.exp((best_value - new_value) / temperature):
                best_value = new_value
                best_index = new_point

        # Return the optimized value
        return best_value