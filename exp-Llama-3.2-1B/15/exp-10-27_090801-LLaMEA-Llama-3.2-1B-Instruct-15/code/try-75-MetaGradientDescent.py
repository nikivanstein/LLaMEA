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

    def select_solution(self):
        """
        Select a new individual based on the current solution and the probability of refinement.

        Returns:
            tuple: A tuple containing the new individual and the probability of refinement.
        """
        # Calculate the fitness of the current solution
        fitness = self.evaluate_fitness(self.param_values)

        # Select a new individual based on the current solution and the probability of refinement
        new_individual = self.evaluate_fitness(self.param_values + self.noise * np.random.normal(0, 1, self.dim))
        probability_of_refinement = 0.15
        if fitness < new_individual:
            new_individual, probability_of_refinement = self.evaluate_fitness(self.param_values + self.noise * np.random.normal(0, 1, self.dim)), probability_of_refinement

        return new_individual, probability_of_refinement

    def mutate(self, individual):
        """
        Mutate the current solution by changing one parameter value with a small probability.

        Args:
            individual (float): The current individual.

        Returns:
            float: The mutated individual.
        """
        # Calculate the probability of mutation
        mutation_probability = 0.01

        # Mutate the current solution
        mutated_individual = individual + np.random.normal(0, 1, self.dim) * np.random.uniform(-5.0, 5.0, self.dim)
        if random.random() < mutation_probability:
            mutated_individual -= np.random.normal(0, 1, self.dim) * np.random.uniform(-5.0, 5.0, self.dim)

        return mutated_individual

# Description: Novel Metaheuristic Algorithm for Black Box Optimization on BBOB Test Suite
# Code: 