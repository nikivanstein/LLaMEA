import numpy as np
import random
import copy

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

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def select_next_individual(self, current_individual, current_fitness):
        """
        Select the next individual based on the current fitness and a novel heuristic strategy.

        Args:
            current_individual (numpy array): The current individual.
            current_fitness (float): The current fitness of the individual.

        Returns:
            tuple: The next individual and its fitness.
        """
        # Calculate the average fitness of the current individual
        avg_fitness = current_fitness / (current_individual.shape[0] + 1)

        # Select the individual with the highest average fitness
        next_individual = np.random.choice(current_individual.shape[0], p=avg_fitness)

        # Refine the strategy by changing the direction of the selected individual
        direction = np.random.choice([-1, 1], size=dim)
        next_individual = np.where(direction > 0, current_individual[next_individual], current_individual[next_individual])

        # Update the fitness of the selected individual
        next_fitness = func(next_individual)

        return next_individual, next_fitness

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 