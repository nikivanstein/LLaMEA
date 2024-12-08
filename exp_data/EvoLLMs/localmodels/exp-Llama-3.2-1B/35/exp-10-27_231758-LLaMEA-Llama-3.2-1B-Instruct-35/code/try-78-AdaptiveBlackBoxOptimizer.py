import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
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

    def adaptive(self, func, bounds, initial_guess, budget):
        """
        Adaptive Black Box Optimization using Differential Evolution.

        Args:
        func (function): The black box function to optimize.
        bounds (list): The search space bounds for each dimension.
        initial_guess (list): The initial guess for each dimension.
        budget (int): The maximum number of function evaluations.

        Returns:
        tuple: A tuple containing the optimized function values and the updated bounds.
        """
        # Refine the strategy using probability 0.35
        for _ in range(int(np.ceil(budget / 5))):
            # Sample new function values
            new_func_values = np.random.uniform(bounds[0], bounds[1], self.dim)

            # Evaluate the new function values
            new_func_values_evals = np.sum(np.abs(func(new_func_values)))

            # Update the bounds if the new function values are better
            if new_func_values_evals < np.sum(np.abs(self.func_values)):
                self.func_values = new_func_values
                self.func_evals = 0
                for dim, value in zip(self.func_values, new_func_values):
                    bounds[dim] = (bounds[dim] - 1) / 5 + 1
        return self.func_values, bounds

# Description: Adaptive Black Box Optimization using Differential Evolution.
# Code: 