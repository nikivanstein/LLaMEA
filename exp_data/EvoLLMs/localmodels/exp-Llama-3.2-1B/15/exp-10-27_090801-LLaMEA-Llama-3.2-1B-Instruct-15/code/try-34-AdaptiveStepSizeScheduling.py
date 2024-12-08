import numpy as np
import random

class AdaptiveStepSizeScheduling:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the adaptive step size scheduling algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0
        self.step_size = 1.0

    def __call__(self, func):
        """
        Optimize the black box function `func` using adaptive step size scheduling.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the parameter values to random values within the search space
        self.param_values = np.random.uniform(-5.0, 5.0, self.dim)

        # Accumulate noise in the objective function evaluations
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise
            self.param_values += self.noise * np.random.normal(0, 1, self.dim)

            # Schedule the step size for the next iteration
            self.step_size = np.sqrt(np.max(np.abs(func_value - np.min(func_value)))) * 0.1

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

# Example usage:
budget = 100
dim = 10
noise_level = 0.1
algorithm = AdaptiveStepSizeScheduling(budget, dim, noise_level)
func = lambda x: np.sin(x)
optimized_individual, objective_value = algorithm(func)
print(f"Optimized individual: {optimized_individual}")
print(f"Objective value: {objective_value}")