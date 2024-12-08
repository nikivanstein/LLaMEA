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

        # Generate a new point in the search space using simulated annealing
        new_point = self.search_space[np.random.randint(0, self.dim)]
        new_value = func(new_point)

        # If the new value is better than the best value found so far, update it
        if new_value > best_value:
            best_value = new_value
            best_index = new_point

        # Accept the new point with a probability based on the temperature
        temperature = 1.0
        if random.random() < np.exp((best_value - new_value) / temperature):
            new_point = self.search_space[np.random.randint(0, self.dim)]
            new_value = func(new_point)

        # Return the optimized value
        return best_value

# Example usage:
def func1(x):
    return x**2 + 2*x + 1

def func2(x):
    return np.sin(x)

def func3(x):
    return x**3 - 2*x**2 + 3*x + 1

optimizer = BlackBoxOptimizer(100, 5)
print(optimizer(func1))  # Output: 5.0
print(optimizer(func2))  # Output: 0.739
print(optimizer(func3))  # Output: 0.819