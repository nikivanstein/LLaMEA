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

    def __call__(self, func, budget=100, mutation_rate=0.1):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.
            budget (int, optional): The maximum number of function evaluations allowed. Defaults to 100.
            mutation_rate (float, optional): The probability of mutation. Defaults to 0.1.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Perform the specified number of function evaluations
        for _ in range(min(budget, self.budget)):
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
            # Generate a new random point in the search space
            point = self.search_space[np.random.randint(0, self.dim)]

            # Evaluate the function at the new point
            value = func(point)

            # If the new value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > best_value:
                best_value = value
                best_index = point

        # Return the optimized value
        return best_value

# Example usage:
# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# ```python
# ```python
optimizer = BlackBoxOptimizer(1000, 10)
best_func = lambda x: x**2
best_value = optimizer(best_func)
print(f"Best function value: {best_value}")

# Refine the strategy
optimizer = BlackBoxOptimizer(1000, 10)
best_func = lambda x: x**2 + 0.1*x
best_value = optimizer(best_func)
print(f"Refined best function value: {best_value}")

# Perform mutation
optimizer = BlackBoxOptimizer(1000, 10)
best_func = lambda x: x**2 + 0.1*x
best_value = optimizer(best_func)
print(f"Refined best function value after mutation: {best_value}")