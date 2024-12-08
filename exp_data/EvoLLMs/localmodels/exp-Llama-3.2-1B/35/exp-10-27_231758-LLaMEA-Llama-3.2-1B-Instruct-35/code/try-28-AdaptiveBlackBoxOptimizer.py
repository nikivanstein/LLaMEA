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

    def adaptive_black_box(self, func, bounds, init, max_iter=100, tol=1e-6):
        """
        Adaptive Black Box Optimization using Differential Evolution.

        Parameters:
        func (function): The black box function to optimize.
        bounds (list): The search space bounds for each dimension.
        init (list): The initial population for each dimension.
        max_iter (int): The maximum number of iterations.
        tol (float): The tolerance for convergence.

        Returns:
        list: The optimized solution.
        """
        # Initialize the population with random values
        pop = init * np.ones((self.dim, self.budget))
        
        # Run the optimization for a fixed number of iterations
        for _ in range(max_iter):
            # Evaluate the function at each individual in the population
            func_values = np.array([func(pop_i) for pop_i in pop])
            
            # Select the fittest individuals
            idx = np.argmin(np.abs(func_values))
            pop[idx] = init[idx]
            
            # Update the bounds
            for i in range(self.dim):
                if pop[i, idx] < bounds[i][0]:
                    bounds[i][0] = bounds[i][0] + (bounds[i][1] - bounds[i][0]) / 10
                elif pop[i, idx] > bounds[i][1]:
                    bounds[i][1] = bounds[i][1] - (bounds[i][1] - bounds[i][0]) / 10
        
        # Return the fittest individual
        return pop[np.argmin(np.abs(func_values))]

# Description: Adaptive Black Box Optimization using Differential Evolution
# Code: 
# ```python
# AdaptiveBlackBoxOptimizer(budget, dim).adaptive_black_box(func, bounds, init, max_iter=100, tol=1e-6)