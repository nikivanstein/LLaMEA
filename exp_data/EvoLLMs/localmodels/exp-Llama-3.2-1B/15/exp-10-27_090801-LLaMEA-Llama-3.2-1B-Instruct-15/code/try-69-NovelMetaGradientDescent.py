import numpy as np
import random
import os

class NovelMetaGradientDescent:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the novel meta-gradient descent algorithm.

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
        Optimize the black box function `func` using novel meta-gradient descent.

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

        # Refine the solution by changing the individual lines of the selected solution
        self.param_values = np.random.uniform(-5.0, 5.0, self.dim)
        self.param_values[0] = self.param_values[0] + 0.2 * (self.param_values[0] - 0.5)
        self.param_values[1] = self.param_values[1] + 0.2 * (self.param_values[1] - 0.5)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def update(self, func, new_individual, new_fitness):
        """
        Update the algorithm with a new individual and fitness value.

        Args:
            func (callable): The black box function to evaluate.
            new_individual (array): The new individual to evaluate.
            new_fitness (float): The new fitness value.
        """
        # Refine the solution by changing the individual lines of the selected solution
        self.param_values = np.random.uniform(-5.0, 5.0, self.dim)
        self.param_values[0] = self.param_values[0] + 0.2 * (self.param_values[0] - 0.5)
        self.param_values[1] = self.param_values[1] + 0.2 * (self.param_values[1] - 0.5)
        self.param_values[2] = self.param_values[2] + 0.2 * (self.param_values[2] - 0.5)
        self.param_values[3] = self.param_values[3] + 0.2 * (self.param_values[3] - 0.5)

        # Evaluate the objective function with the new individual and fitness value
        new_func_value = func(new_individual)

        # Update the fitness value
        new_fitness = new_func_value

        # Update the algorithm's score
        if new_fitness > self.f(new_fitness, self.param_values):
            self.f(new_fitness, self.param_values) = new_fitness

        # Save the updated algorithm's parameters to a file
        if not os.path.exists("currentexp"):
            os.makedirs("currentexp")
        np.save("currentexp/aucs-" + str(self.f(new_fitness, self.param_values)), new_fitness)

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 