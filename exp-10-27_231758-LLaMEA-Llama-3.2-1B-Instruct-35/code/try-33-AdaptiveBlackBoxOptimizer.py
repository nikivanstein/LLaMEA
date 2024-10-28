import numpy as np
import random
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

    def optimize(self, func):
        """
        Optimize the black box function using the AdaptiveBlackBoxOptimizer.

        Parameters:
        func (function): The black box function to optimize.

        Returns:
        tuple: A tuple containing the optimized function value, the optimized function value at the optimal point, and the number of evaluations required.
        """
        # Refine the search space using a probabilistic approach
        while True:
            # Generate a new point within the current search space
            new_point = np.random.uniform(-5.0, 5.0, self.dim)

            # Evaluate the new point using the original function
            new_func_value = func(new_point)

            # Generate a random perturbation of the new point
            perturbation = np.random.uniform(-0.1, 0.1, self.dim)

            # Evaluate the perturbed point using the original function
            perturbed_func_value = func(new_point + perturbation)

            # Calculate the probability of accepting the new point
            prob = np.abs(new_func_value - perturbed_func_value) / np.abs(new_func_value - func(new_point))

            # Accept the new point with a probability based on the probability of accepting
            if random.random() < prob:
                return new_func_value, new_func_value, self.func_evals + 1

            # If the new point is not accepted, move back to the previous point
            self.func_values = np.roll(self.func_values, -1)
            self.func_evals -= 1

# Description: AdaptiveBlackBoxOptimizer
# Code: 