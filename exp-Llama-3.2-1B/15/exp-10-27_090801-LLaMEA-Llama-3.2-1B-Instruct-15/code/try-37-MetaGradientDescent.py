import numpy as np
import random

class MetaGradientDescent:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-gradient descent algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0
        self.best_individual = None

    def __call__(self, func):
        """
        Optimize the black box function `func` using meta-gradient descent.

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

        # Evaluate the objective function for the best individual found so far
        if self.best_individual is None or np.linalg.norm(self.param_values - self.best_individual) < 1e-6:
            self.best_individual = self.param_values
            self.best_fitness = func(self.best_individual)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func(self.best_individual)

    def mutate(self, individual):
        """
        Mutate the individual to refine its strategy.

        Args:
            individual (numpy array): The individual to mutate.

        Returns:
            numpy array: The mutated individual.
        """
        # Randomly select a new individual within the search space
        new_individual = individual + np.random.uniform(-5.0, 5.0, self.dim)

        # Check if the new individual is within the search space
        if np.linalg.norm(new_individual - individual) > 1e-6:
            raise ValueError("New individual is outside the search space")

        return new_individual

# One-line description with the main idea
# MetaGradientDescent: A novel metaheuristic algorithm that combines meta-learning and gradient descent to optimize black box functions.

# Code: