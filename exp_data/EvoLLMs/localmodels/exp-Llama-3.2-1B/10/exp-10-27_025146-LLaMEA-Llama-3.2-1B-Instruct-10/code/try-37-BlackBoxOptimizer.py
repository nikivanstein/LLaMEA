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

        # Perform simulated annealing to refine the strategy
        temperature = 1000
        for _ in range(1000):
            # Generate a new point in the search space using the current best point
            new_point = self.search_space[np.random.randint(0, self.dim)]

            # Evaluate the function at the new point
            new_value = func(new_point)

            # If the new value is better than the current best value, update the best value
            if new_value > best_value:
                best_value = new_value

            # Calculate the probability of accepting the new point
            probability = np.exp((best_value - new_value) / temperature)

            # Accept the new point with a probability less than 1
            if random.random() < probability:
                best_index = new_point
                break

        # Return the optimized value
        return best_value

# Example usage:
def func1(point):
    return np.sum(point**2)

def func2(point):
    return np.prod(point)

optimizer = BlackBoxOptimizer(100, 5)
print(optimizer(func1))  # Output: optimized value of func1
print(optimizer(func2))  # Output: optimized value of func2