import random
import numpy as np
from scipy.optimize import differential_evolution

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

    def novelty_metaheuristic(self, func, budget):
        """
        Novel Metaheuristic Algorithm for Black Box Optimization.

        Description: This algorithm uses differential evolution to optimize the black box function.
        The novelty heuristic strategy involves using the best solution found so far to refine the search space.

        Code: 
        ```python
        import numpy as np
        from scipy.optimize import differential_evolution

        def novelty_metaheuristic(func, budget):
            # Initialize the best value and its corresponding index
            best_value = float('-inf')
            best_index = -1

            # Perform the specified number of function evaluations
            for _ in range(budget):
                # Generate a random point in the search space
                point = self.search_space[np.random.randint(0, self.dim)]

                # Evaluate the function at the current point
                value = func(point)

                # If the current value is better than the best value found so far,
                # update the best value and its corresponding index
                if value > best_value:
                    best_value = value
                    best_index = point

            # Refine the search space using the best solution found so far
            refined_search_space = np.linspace(-5.0, 5.0, best_index.shape[0])
            best_point = np.argmax(func(refined_search_space))

            # Return the optimized value
            return best_value, best_point

        # Return the novelty metaheuristic algorithm
        return novelty_metaheuristic