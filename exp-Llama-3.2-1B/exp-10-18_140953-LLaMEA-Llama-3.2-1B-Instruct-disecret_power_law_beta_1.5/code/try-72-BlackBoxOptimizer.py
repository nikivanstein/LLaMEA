# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim, learning_rate=0.01, exploration_rate=0.1, exploration_threshold=0.5):
        """
        Initializes the BlackBoxOptimizer with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
            learning_rate (float, optional): The learning rate for the metaheuristic algorithm. Defaults to 0.01.
            exploration_rate (float, optional): The exploration rate for the metaheuristic algorithm. Defaults to 0.1.
            exploration_threshold (float, optional): The exploration threshold for the metaheuristic algorithm. Defaults to 0.5.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        """
        Evaluates the given black box function using the given budget and returns the optimized result.

        Args:
            func (function): The black box function to optimize.

        Returns:
            float: The optimized result of the function.
        """
        # Initialize the current point in the search space
        current_point = self.search_space.copy()

        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Generate a new point in the search space
            new_point = current_point + np.random.uniform(-1.0, 1.0, self.dim)

            # Evaluate the function at the new point
            func_value = func(new_point)

            # If the function value is better than the current best, update the current point
            if func_value < np.max(current_point):
                # Use the current point as the next point if it's better or if it's within the exploration threshold
                if np.random.rand() < exploration_rate:
                    current_point = new_point
                # Otherwise, use the current point as the next point if it's better
                else:
                    current_point = new_point

        # Return the optimized result
        return current_point

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 