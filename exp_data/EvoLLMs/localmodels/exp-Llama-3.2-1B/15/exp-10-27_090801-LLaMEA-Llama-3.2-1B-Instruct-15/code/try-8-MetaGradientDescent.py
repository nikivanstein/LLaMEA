# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
import random
from scipy.optimize import differential_evolution

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

    def optimize(self, func, budget, dim, noise_level=0.1):
        """
        Optimize the black box function `func` using meta-gradient descent.

        Args:
            func (callable): The black box function to optimize.
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the population of individuals with random parameter values
        population = [np.random.uniform(-5.0, 5.0, dim) for _ in range(100)]

        # Evolve the population using meta-gradient descent
        for _ in range(1000):
            # Evaluate the fitness of each individual
            fitness = [self.__call__(func, individual) for individual in population]

            # Select the fittest individuals
            fittest_individuals = population[np.argsort(fitness)]

            # Select the next generation based on the probability of refinement
            next_generation = [fittest_individuals[i] + self.noise * np.random.normal(0, 1, dim) for i in range(100)]

            # Replace the old population with the new generation
            population = next_generation

        # Return the fittest individual and its fitness
        return population[0], fitness[0]

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

meta_gradient_descent = MetaGradientDescent(1000, 2, noise_level=0.1)
optimized_individual, fitness = meta_gradient_descent.optimize(func, 1000, 2, noise_level=0.1)

# Save the optimized individual and fitness to a file
np.save("currentexp/aucs-MetaGradientDescent-0.npy", (optimized_individual, fitness))