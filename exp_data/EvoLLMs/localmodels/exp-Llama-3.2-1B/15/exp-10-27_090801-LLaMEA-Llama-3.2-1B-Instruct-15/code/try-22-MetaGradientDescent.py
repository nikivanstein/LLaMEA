# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
import random
from scipy.optimize import minimize

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

    def meta_gradient_descent(self, func, bounds, initial_point, noise_level):
        """
        Metaheuristic algorithm to optimize a black box function.

        Args:
            func (callable): The black box function to optimize.
            bounds (list): The search space bounds.
            initial_point (tuple): The initial parameter values.
            noise_level (float): The level of noise accumulation.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the population with the initial point
        population = [initial_point]

        # Evolve the population using meta-gradient descent
        for _ in range(100):  # Evolve for 100 generations
            # Evaluate the fitness of each individual
            fitness_values = [func(individual, bounds) for individual in population]

            # Select the fittest individuals
            fittest_individuals = [individual for _, individual in sorted(zip(fitness_values, population), reverse=True)]

            # Select two parents using tournament selection
            parents = [fittest_individuals[0], fittest_individuals[1]]

            # Mutate the parents
            for _ in range(10):  # Mutate 10% of the parents
                parent1, parent2 = random.sample(parents, 2)
                parent1, parent2 = parent1, parent2
                parent1 = self.meta_gradient_descent(func, bounds, parent1, noise_level)
                parent2 = self.meta_gradient_descent(func, bounds, parent2, noise_level)

                # Crossover the parents to create a new individual
                child = (0.5 * (parent1 + parent2),)

                # Mutate the child
                for _ in range(10):  # Mutate 10% of the child
                    child = (child[0] + np.random.normal(0, 1, self.dim) * np.random.normal(0, 1, self.dim),)

                # Add the child to the population
                population.append(child)

        # Return the fittest individual
        return population[0]

    def evaluate_fitness(self, func, bounds, initial_point):
        """
        Evaluate the fitness of a given individual.

        Args:
            func (callable): The black box function to evaluate.
            bounds (list): The search space bounds.
            initial_point (tuple): The initial parameter values.

        Returns:
            float: The fitness value of the individual.
        """
        return func(initial_point, bounds)

# Example usage:
def func1(x):
    return np.sum(x**2)

bounds = [(-5, 5), (-5, 5)]

initial_point = (-5, -5)
meta_gradient_descent = MetaGradientDescent(100, 2, noise_level=0.1)

optimized_point = meta_gradient_descent.meta_gradient_descent(func1, bounds, initial_point, noise_level=0.1)
optimized_point, _ = meta_gradient_descent.meta_gradient_descent(func1, bounds, optimized_point, noise_level=0.1)

# Print the optimized point
print(optimized_point)