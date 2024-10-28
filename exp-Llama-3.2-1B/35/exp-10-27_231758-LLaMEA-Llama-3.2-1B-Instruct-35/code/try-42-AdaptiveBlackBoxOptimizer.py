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

    def adaptive_black_box(self, func, bounds, initial_guess, budget):
        """
        Adaptive Black Box Optimization using Differential Evolution.

        Parameters:
        func (function): The black box function to optimize.
        bounds (list): A list of tuples specifying the lower and upper bounds for each dimension.
        initial_guess (list): The initial guess for the optimization.
        budget (int): The maximum number of function evaluations allowed.

        Returns:
        tuple: A tuple containing the optimized function value, the optimized function, and the number of evaluations made.
        """
        # Refine the search space using probability 0.35
        bounds = [(min(bounds[i][0], bounds[i][1]), max(bounds[i][0], bounds[i][1])) for i in range(len(bounds))]
        initial_guess = np.array(initial_guess) / np.sqrt(len(initial_guess))  # Normalize the initial guess

        # Run Differential Evolution
        result = differential_evolution(lambda x: -func(x), bounds, initial_guess=initial_guess, x0=initial_guess, popsize=100, ngen=50, tol=1e-6, maxiter=100, verbose=True)

        # Refine the search space using probability 0.35
        bounds = [(min(bounds[i][0], bounds[i][1]), max(bounds[i][0], bounds[i][1])) for i in range(len(bounds))]
        return result.fun, func(result.x), result.iterations


# Description: Adaptive Black Box Optimization using Differential Evolution
# Code: 