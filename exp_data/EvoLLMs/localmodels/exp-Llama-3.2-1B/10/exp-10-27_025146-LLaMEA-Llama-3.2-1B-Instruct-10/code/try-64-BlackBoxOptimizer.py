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

        # Return the optimized value
        return best_value

class BlackBoxOptimizerMetaheuristic:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizerMetaheuristic with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func, mutation_rate):
        """
        Optimize the black box function using the BlackBoxOptimizerMetaheuristic.

        Args:
            func (callable): The black box function to optimize.
            mutation_rate (float): The probability of applying a mutation to the current individual.

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

            # Apply mutation to the current individual
            if np.random.rand() < mutation_rate:
                new_index = np.random.randint(0, self.dim)
                new_point = self.search_space[new_index]
                new_value = func(new_point)

                # Evaluate the new function value
                new_value = func(new_point)

                # If the new value is better than the best value found so far,
                # update the best value and its corresponding index
                if new_value < best_value:
                    best_value = new_value
                    best_index = new_index

        # Return the optimized value
        return best_value

# Example usage
def sphere(x):
    return sum(x[i]**2 for i in range(x.shape[0]))

def bobyfark(x):
    return sum(x[i] for i in range(x.shape[0]))

optimizer = BlackBoxOptimizerMetaheuristic(budget=100, dim=5)
best_solution = optimizer(BlackBoxOptimizer(sphere), mutation_rate=0.1)

print("Best solution:", best_solution)
print("Best value:", best_solution)