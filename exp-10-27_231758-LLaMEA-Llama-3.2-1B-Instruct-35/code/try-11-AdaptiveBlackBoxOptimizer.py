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

    def adaptive_black_box(self, func, bounds, initial_guess, max_iter=100, tol=1e-6):
        """
        Adaptive Black Box Optimization using Differential Evolution.

        Args:
        func (function): The objective function to optimize.
        bounds (list): The bounds for each dimension.
        initial_guess (list): The initial guess for each dimension.
        max_iter (int): The maximum number of iterations.
        tol (float): The tolerance for convergence.

        Returns:
        dict: A dictionary containing the optimized solution, its score, and the number of evaluations.
        """
        # Refine the strategy based on the current population
        if len(self.func_values) > 10:
            # If the population is large, refine the strategy
            new_initial_guess = np.random.uniform(bounds)
            new_bounds = [bounds[i] + (bounds[i] - bounds[i-1]) * 0.1 for i in range(1, len(bounds))]
            new_func_values = np.zeros(len(new_bounds))
            for _ in range(100):
                func_values = np.array([func(new_initial_guess[i]) for i in range(len(new_initial_guess))])
                idx = np.argmin(np.abs(func_values))
                new_initial_guess[idx] = new_initial_guess[i]
                new_bounds[idx] = new_bounds[i] + (new_bounds[i] - new_bounds[i-1]) * 0.1
                new_func_values[idx] = func(new_initial_guess[idx])
                if np.abs(func_values[idx] - func(new_initial_guess[idx])) < tol:
                    break
            self.func_values = new_func_values
            self.func_values = np.array([self.func_values[i] for i in range(len(self.func_values))])
            self.dim = len(new_bounds)
            self.func_evals = 0
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        return {
            'optimized_solution': np.array(self.func_values).reshape(-1, self.dim),
           'score': self.func_values.mean(),
            'num_evaluations': self.func_evals,
            'iter_count': max_iter
        }

# Description: Adaptive Black Box Optimization using Differential Evolution
# Code: 