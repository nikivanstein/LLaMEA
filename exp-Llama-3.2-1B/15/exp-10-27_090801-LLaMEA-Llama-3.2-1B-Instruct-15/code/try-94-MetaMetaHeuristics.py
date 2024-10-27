import numpy as np
import random
import os

class MetaMetaHeuristics:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-metaheuristic algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0
        self.current_exp = {}
        self.exp_name = "MetaMetaHeuristics"

    def __call__(self, func):
        """
        Optimize the black box function `func` using meta-metaheuristics.

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

        # Update the current experiment
        self.current_exp[self.exp_name] = (self.param_values, func_value)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def update(self):
        """
        Update the current experiment.
        """
        # Check if the experiment has been evaluated before
        if self.current_exp:
            # Get the current experiment
            current_exp = self.current_exp[self.exp_name]

            # Refine the strategy based on the current experiment
            self.noise = 0.15 * current_exp[1] / np.mean([current_exp[1] for current_exp in self.current_exp.values()])
            self.param_values = np.random.uniform(-5.0, 5.0, self.dim)

            # Update the current experiment
            self.current_exp[self.exp_name] = (self.param_values, current_exp[1])

    def evaluate_fitness(self, func):
        """
        Evaluate the fitness of the current experiment.

        Args:
            func (callable): The black box function to evaluate.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Update the current experiment
        self.update()

        # Evaluate the fitness of the current experiment
        updated_individual = self.evaluate_new_individual(func)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func(updated_individual)

    def evaluate_new_individual(self, func):
        """
        Evaluate a new individual using the current experiment.

        Args:
            func (callable): The black box function to evaluate.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Refine the strategy based on the current experiment
        self.noise = 0.15 * np.mean([func(self.param_values + self.noise * np.random.normal(0, 1, self.dim)) for self.param_values in self.current_exp.values()])

        # Evaluate the fitness of the new individual
        new_individual = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

        # Return the optimized parameter values and the objective function value
        return self.param_values, new_individual

# Description: MetaMetaHeuristics for Black Box Optimization
# Code: 