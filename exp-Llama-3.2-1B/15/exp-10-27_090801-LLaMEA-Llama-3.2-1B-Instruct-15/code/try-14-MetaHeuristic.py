# Description: Refining the meta-heuristic algorithm for black box optimization using a combination of exploration-exploitation trade-off and adaptive mutation.
# Code: 
# ```python
import numpy as np
import random

class MetaHeuristic:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-heuristic algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0

    def __call__(self, func):
        """
        Optimize the black box function `func` using meta-heuristic.

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

            # Update the noise level based on the exploration-exploitation trade-off
            if random.random() < 0.15:
                self.noise += 0.01 * np.random.normal(0, 1, self.dim)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def adaptMutation(self, func, param_values, noise):
        """
        Adapt the mutation strategy based on the fitness value.

        Args:
            func (callable): The black box function to optimize.
            param_values (numpy array): The optimized parameter values.
            noise (float): The current noise level.
        """
        # Calculate the fitness value
        fitness = func(param_values)

        # Update the noise level based on the fitness value
        if fitness > 0.5:
            self.noise -= 0.01 * np.random.normal(0, 1, self.dim)
            if self.noise < -0.5:
                self.noise = -0.5
        else:
            self.noise += 0.01 * np.random.normal(0, 1, self.dim)
            if self.noise > 0.5:
                self.noise = 0.5

# Example usage:
meta_heuristic = MetaHeuristic(1000, 10)
func = lambda x: x**2
optimized_values, _ = meta_heuristic(func)