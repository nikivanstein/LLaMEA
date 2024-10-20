import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a given budget and dimensionality.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        """
        Optimize a black box function using the BlackBoxOptimizer.

        Args:
        func (function): The black box function to optimize.

        Returns:
        float: The optimized value of the function.
        """
        # Create a copy of the search space to avoid modifying the original
        search_space = self.search_space.copy()

        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Evaluate the function at the current search space point
            func_value = func(search_space)

            # Update the search space point to be the next possible point
            search_space = np.array([self.search_space[:, 0] + np.random.uniform(-1, 1, size=dim),
                                    self.search_space[:, 1] + np.random.uniform(-1, 1, size=dim),
                                    self.search_space[:, 2] + np.random.uniform(-1, 1, size=dim),
                                    self.search_space[:, 3] + np.random.uniform(-1, 1, size=dim)])

        # Return the optimized function value
        return func_value