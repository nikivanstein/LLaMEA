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

    def adaptive_black_box(self, func, budget, dim, iterations=1000, tol=1e-6):
        """
        Adaptive Black Box Optimization using Differential Evolution.

        Parameters:
        func (function): The black box function to optimize.
        budget (int): The number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        iterations (int, optional): The number of iterations for the differential evolution algorithm. Defaults to 1000.
        tol (float, optional): The tolerance for the convergence check. Defaults to 1e-6.

        Returns:
        float: The optimal function value.
        """
        # Initialize the population with random values in the search space
        population = self.func_values[np.random.choice(self.dim, size=100, replace=False)]

        # Run the differential evolution algorithm
        for _ in range(iterations):
            # Evaluate the population using the given function
            func_values = func(population)
            # Check if the population has converged
            if np.all(np.abs(func_values - population) < tol):
                break

            # Refine the population by selecting the individual with the minimum absolute difference
            idx = np.argmin(np.abs(func_values - population))
            population[idx] = func(population[idx])

        # Return the optimal function value
        return func(population)

# Description: Adaptive Black Box Optimization using Differential Evolution
# Code: 