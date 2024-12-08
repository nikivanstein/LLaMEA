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

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

class MetaGradientDescentMetaheuristic:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-gradient descent metaheuristic.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.metaheuristic = MetaGradientDescent(budget, dim, noise_level)

    def __call__(self, func, iterations=1000):
        """
        Optimize the black box function `func` using meta-gradient descent metaheuristic.

        Args:
            func (callable): The black box function to optimize.
            iterations (int, optional): The number of iterations. Defaults to 1000.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the population with random parameter values
        population = [self.metaheuristic.__call__(func, 10) for _ in range(100)]

        # Evolve the population over iterations
        for _ in range(iterations):
            # Select the fittest individual
            fittest_individual = population[np.argmax([individual[1] for individual in population])]

            # Perturb the fittest individual
            perturbed_individual = fittest_individual[:]

            # Evaluate the objective function with perturbed individual
            func_value = func(perturbed_individual + self.metaheuristic.noise_level * np.random.normal(0, 1, self.metaheuristic.dim))

            # Update the population with perturbed individual
            population.append((perturbed_individual, func_value))

        # Return the fittest individual and the objective function value
        return population[0], population[0][1]

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 