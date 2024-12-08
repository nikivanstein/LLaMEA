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

        # Refine the strategy by changing the individual lines of the selected solution
        for i in range(self.budget):
            # Select a new individual with a probability based on the current fitness
            new_individual = self.evaluate_fitness(self.param_values)
            if random.random() < 0.15:
                new_individual = func_value * np.random.normal(0, 1, self.dim)

            # Replace the current individual with the new one
            self.param_values = new_individual

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of the individual.

        Args:
            individual (numpy array): The individual to evaluate.

        Returns:
            float: The fitness of the individual.
        """
        # Evaluate the objective function at the individual
        func_value = self.func(individual)

        # Return the fitness as a tuple
        return func_value, individual

    def func(self, individual):
        """
        Evaluate the objective function at the individual.

        Args:
            individual (numpy array): The individual to evaluate.

        Returns:
            float: The value of the objective function at the individual.
        """
        # Define the black box function
        def f(x):
            return np.sum(x**2)

        # Evaluate the objective function
        func_value = f(individual)

        # Return the fitness as a tuple
        return func_value, individual