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

    def select_solution(self, func, budget, dim):
        """
        Select a solution using the probability of 0.15.

        Args:
            func (callable): The black box function to optimize.
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.

        Returns:
            tuple: A tuple containing the selected individual and the objective function value.
        """
        # Initialize the population with random solutions
        population = [self.evaluate_fitness(func, i) for i in range(budget)]

        # Select the solution with probability 0.15
        selected_individual = random.choices(population, weights=[1 if i in population else 0 for i in population])[0]

        # Update the population with the selected individual
        new_population = [self.evaluate_fitness(func, i) for i in population]
        new_population = [i if i < selected_individual else 1 - i for i in new_population]
        population = new_population

        # Return the selected individual and the objective function value
        return selected_individual, func(selected_individual)

    def mutate(self, func, population, dim):
        """
        Mutate the population using the probability of 0.15.

        Args:
            func (callable): The black box function to optimize.
            population (list): The population of individuals.
            dim (int): The dimensionality of the problem.

        Returns:
            list: The mutated population.
        """
        # Initialize the mutated population with random solutions
        mutated_population = [self.evaluate_fitness(func, i) for i in range(len(population))]

        # Select the individual with probability 0.15
        selected_individual = random.choices(mutated_population, weights=[1 if i in mutated_population else 0 for i in mutated_population])[0]

        # Mutate the selected individual
        mutated_individual = self.evaluate_fitness(func, selected_individual)
        mutated_population = [i if i < selected_individual else 1 - i for i in mutated_population]

        # Return the mutated population
        return mutated_population

# Description: MetaGradientDescent algorithm with adaptive mutation probability.
# Code: 