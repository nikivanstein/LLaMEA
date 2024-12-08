import numpy as np
import random

class MetaHeuristicOptimization:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-heuristic optimization algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0

    def __call__(self, func, num_evaluations):
        """
        Optimize the black box function `func` using meta-heuristic optimization.

        Args:
            func (callable): The black box function to optimize.
            num_evaluations (int): The maximum number of function evaluations allowed.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the parameter values to random values within the search space
        self.param_values = np.random.uniform(-5.0, 5.0, self.dim)

        # Accumulate noise in the objective function evaluations
        for _ in range(num_evaluations):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise
            self.param_values += self.noise * np.random.normal(0, 1, self.dim)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func(self.param_values)

    def select_solution(self, func, num_evaluations):
        """
        Select a solution from the current population based on the probability of refinement.

        Args:
            func (callable): The black box function to optimize.
            num_evaluations (int): The maximum number of function evaluations allowed.

        Returns:
            tuple: A tuple containing the selected solution and the number of evaluations.
        """
        # Initialize the current population with random solutions
        current_population = [self.__call__(func, 100) for _ in range(100)]

        # Select the solution with the highest fitness value
        selected_solution = max(current_population, key=lambda x: x[0])

        # Refine the selected solution based on the probability of refinement
        if random.random() < 0.15:
            selected_solution = self.__call__(func, num_evaluations)
            num_evaluations -= 1

        return selected_solution, num_evaluations

# Description: Meta-Heuristic Optimization Algorithm for Black Box Optimization
# Code: 