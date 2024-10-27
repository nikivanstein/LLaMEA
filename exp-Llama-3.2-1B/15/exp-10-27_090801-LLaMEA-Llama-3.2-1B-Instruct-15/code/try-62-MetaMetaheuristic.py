import numpy as np
import random

class MetaMetaheuristic:
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

    def __call__(self, func, iterations=100):
        """
        Optimize the black box function `func` using meta-heuristic search.

        Args:
            func (callable): The black box function to optimize.
            iterations (int, optional): The number of iterations. Defaults to 100.

        Returns:
            tuple: A tuple containing the optimized parameter values, the objective function value, and a dictionary with the number of evaluations and the average fitness.
        """
        # Initialize the parameter values to random values within the search space
        self.param_values = np.random.uniform(-5.0, 5.0, self.dim)

        # Initialize the fitness dictionary
        fitness_dict = {"evaluations": 0, "average_fitness": 0.0}

        # Accumulate noise in the objective function evaluations
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise
            self.param_values += self.noise * np.random.normal(0, 1, self.dim)

            # Calculate the fitness
            fitness = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the fitness dictionary
            fitness_dict["evaluations"] += 1
            fitness_dict["average_fitness"] = (fitness_dict["average_fitness"] + fitness) / 2

        # Return the optimized parameter values, the objective function value, and the fitness dictionary
        return self.param_values, func_value, fitness_dict

# One-line description: Novel metaheuristic algorithm that uses a combination of mutation and selection to refine its strategy
# Code: 