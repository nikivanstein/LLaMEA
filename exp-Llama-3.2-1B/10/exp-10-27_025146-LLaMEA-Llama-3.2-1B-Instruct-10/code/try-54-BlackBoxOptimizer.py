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

            # Apply a series of small perturbations to the current individual
            perturbation = np.random.uniform(-0.1, 0.1, self.dim)
            new_individual = point + perturbation

            # Evaluate the function at the new individual
            new_value = func(new_individual)

            # If the new value is better than the best value found so far,
            # accept the new individual with a probability based on the temperature
            if new_value > best_value:
                new_value = best_value
                new_index = best_index
            else:
                new_index = np.random.randint(0, self.dim)

            # Update the best value and its corresponding index
            if new_value > best_value:
                best_value = new_value
                best_index = new_index

        # Return the optimized value
        return best_value

# Example usage:
# Create a new BlackBoxOptimizer with a budget of 1000 evaluations and a dimensionality of 5
optimizer = BlackBoxOptimizer(1000, 5)

# Define a black box function to optimize
def func(x):
    return x**2 + 2*x + 1

# Optimize the function using the BlackBoxOptimizer
best_value = optimizer(func)

# Print the optimized value
print(f"Optimized value: {best_value}")