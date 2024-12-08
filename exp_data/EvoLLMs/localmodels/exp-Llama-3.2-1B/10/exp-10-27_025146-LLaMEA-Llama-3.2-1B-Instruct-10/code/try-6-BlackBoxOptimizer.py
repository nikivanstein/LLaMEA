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

    def __call__(self, func, mutation_prob=0.1, cooling_rate=0.99):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.
            mutation_prob (float, optional): The probability of mutating an individual. Defaults to 0.1.
            cooling_rate (float, optional): The rate at which the algorithm cools down. Defaults to 0.99.

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
        if random.random() < mutation_prob:
            # Generate a random mutation
            mutation = np.random.uniform(-1, 1, self.dim)

            # Evaluate the function at the mutated point
            mutated_value = func(point + mutation)

            # If the mutated value is better than the best value found so far,
            # update the best value and its corresponding index
            if mutated_value > best_value:
                best_value = mutated_value
                best_index = point + mutation

        # Return the optimized value
        return best_value

# Example usage:
# 
# Create a new BlackBoxOptimizer with a budget of 100 evaluations and a dimensionality of 5
optimizer = BlackBoxOptimizer(100, 5)

# Define a black box function
def func(x):
    return np.sin(x)

# Optimize the function using the optimizer
optimized_value = optimizer(func, mutation_prob=0.5, cooling_rate=0.9)

# Print the optimized value
print("Optimized value:", optimized_value)