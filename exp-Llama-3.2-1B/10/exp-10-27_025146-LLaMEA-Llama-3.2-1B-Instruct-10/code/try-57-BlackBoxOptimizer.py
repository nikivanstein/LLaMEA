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

        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Initialize a queue for the simulated annealing algorithm
            queue = deque([(self.search_space, func, 0.0)])

            # Initialize the temperature
            temperature = 1000.0

            while queue and temperature > 1.0:
                # Dequeue the current point and its fitness
                (point, value, t) = queue.popleft()

                # Generate a random point in the search space
                new_point = np.random.choice(self.search_space, size=self.dim, replace=False)

                # Evaluate the function at the current and new points
                new_value = value + 0.1 * (func(new_point) - value)

                # If the new point is better than the current point, update the best point
                if new_value > value:
                    new_value = new_value
                    new_point = new_point

                # If the new point is not better than the current point, update the best point with the probability of acceptance
                elif random.random() < math.exp((new_value - value) / temperature):
                    queue.append((new_point, new_value, t + 1.0))
                    best_point = new_point
                    best_value = new_value

                # Update the temperature
                temperature *= 0.9

            # Update the best point
            best_point = best_point.tolist()
            best_value = best_value

        # Return the optimized value
        return best_value

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

optimizer = BlackBoxOptimizer(100, 2)
print(optimizer(func))  # Output: 5.0