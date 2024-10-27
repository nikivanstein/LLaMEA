# Description: Novel Metaheuristic Algorithm for Black Box Optimization (NMABBO)
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

    def __call__(self, func):
        """
        Optimize the black box function using the BlackBoxOptimizerMetaheuristic.

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

class BlackBoxOptimizerMetaheuristicNewton:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizerMetaheuristicNewton with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func):
        """
        Optimize the black box function using the BlackBoxOptimizerMetaheuristicNewton.

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
            # Initialize the gradient of the function
            gradient = np.zeros(self.dim)

            # Iterate over the search space
            for i in range(self.dim):
                # Generate a random point in the search space
                point = self.search_space + np.random.rand() / 10.0

                # Evaluate the function at the current point
                value = func(point)

                # If the current value is better than the best value found so far,
                # update the best value and its corresponding index
                if value > best_value:
                    best_value = value
                    best_index = point

                # Update the gradient of the function
                gradient[i] = (value - best_value) / 0.1

            # Update the best value and its corresponding index
            best_value = np.min(best_value)
            best_index = np.argmin(best_value)

        # Return the optimized value
        return best_value

# Description: Novel Metaheuristic Algorithm for Black Box Optimization (NMABBO)
# Code: 
# ```python
optimizer = BlackBoxOptimizerMetaheuristicNewton(budget=100, dim=5)
print(optimizer(func=lambda x: x**2))