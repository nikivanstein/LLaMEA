import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim, adaptive_search=0.5):
        """
        Initialize the AdaptiveBlackBoxOptimizer with a budget and dimension.

        Args:
            budget (int): Number of function evaluations.
            dim (int): Dimensionality of the search space.
            adaptive_search (float, optional): Probability of switching to adaptive search. Defaults to 0.5.
        """
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.adaptive_search_prob = adaptive_search

    def __call__(self, func):
        """
        Optimize the black box function using the AdaptiveSearch strategy.

        Args:
            func (function): Black box function to optimize.

        Returns:
            None
        """
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

        # Switch to AdaptiveSearch if probability of switching is met
        if np.random.rand() < self.adaptive_search_prob:
            self.__adaptive_search__()

    def __adaptive_search__(self):
        """
        Switch to AdaptiveSearch strategy.

        Returns:
            None
        """
        # Generate a random index within the current search space
        idx = np.random.randint(0, self.dim)

        # Calculate the new function value using the current and new function values
        new_func_value = func(self.func_values[idx]) + np.random.uniform(-1, 1)

        # Update the function values and search space
        self.func_values[idx] = new_func_value
        self.func_values = np.clip(self.func_values, -5.0, 5.0)