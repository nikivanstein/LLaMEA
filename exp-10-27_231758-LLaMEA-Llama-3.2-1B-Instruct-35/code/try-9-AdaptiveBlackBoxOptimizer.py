import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func, alpha=0.6, beta=0.4):
        """
        Optimizes the black box function `func` using a hybrid search strategy.

        Parameters:
        func (function): The black box function to optimize.
        alpha (float, optional): The proportion of function evaluations from the lower bound (default is 0.6).
        beta (float, optional): The proportion of function evaluations from the upper bound (default is 0.4).

        Returns:
        tuple: A tuple containing the optimized function values and the number of function evaluations.
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

        # Refine the strategy using adaptive sampling
        if self.func_evals < self.budget * alpha:
            # Use the upper bound with a higher proportion of evaluations
            self.func_values = np.maximum(self.func_values, func(self.func_values * beta))
        else:
            # Use the lower bound with a lower proportion of evaluations
            self.func_values = np.minimum(self.func_values, func(self.func_values * (1 - beta)))

        return self.func_values, self.func_evals