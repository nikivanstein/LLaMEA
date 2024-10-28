import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim, adaptive_search=False, adaptive_alpha=0.01, adaptive_beta=0.1):
        """
        Initialize the AdaptiveBlackBoxOptimizer.

        Args:
        budget (int): Number of function evaluations.
        dim (int): Dimensionality of the search space.
        adaptive_search (bool, optional): Enable adaptive search strategy. Defaults to False.
        adaptive_alpha (float, optional): Adaptive learning rate for alpha parameter. Defaults to 0.01.
        adaptive_beta (float, optional): Adaptive learning rate for beta parameter. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.alpha = adaptive_alpha
        self.beta = adaptive_beta
        self.search_strategy = adaptive_search

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                if self.search_strategy:
                    idx = np.argmin(np.abs(self.func_values))
                    self.func_values[idx] = func(self.func_values[idx])
                else:
                    idx = np.argmin(np.abs(self.func_values))
                    self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

    def optimize(self, func):
        """
        Optimize the black box function using the AdaptiveBlackBoxOptimizer.

        Args:
        func (function): The black box function to optimize.

        Returns:
        None
        """
        if self.search_strategy:
            alpha = self.alpha / (self.func_evals + 1)
            beta = self.beta / (self.func_evals + 1)
            for _ in range(self.func_evals):
                func(self.func_values)
                self.func_values = np.concatenate((self.func_values, [func(self.func_values[-1])]), axis=0)
                self.func_evals += 1
                if self.func_evals >= self.budget:
                    break
            self.func_values = np.concatenate((self.func_values, [func(self.func_values[-1])]), axis=0)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

# Description: AdaptiveBlackBoxOptimizer with adaptive search strategy
# Code: 
# ```python
# ```python
# AdaptiveBlackBoxOptimizer(1000, 10, adaptive_search=True)
# ```python