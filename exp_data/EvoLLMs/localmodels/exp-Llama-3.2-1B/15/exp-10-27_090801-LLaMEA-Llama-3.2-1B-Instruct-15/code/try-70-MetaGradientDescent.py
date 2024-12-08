import numpy as np
import random
from collections import deque
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
        self.population = deque(maxlen=1000)
        self.fitness_history = deque(maxlen=1000)

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

        # Evaluate the fitness of the current population
        fitness = [func(self.param_values + self.noise * np.random.normal(0, 1, self.dim)) for _ in range(len(self.population))]
        self.fitness_history.extend(fitness)

        # Select the fittest individual
        self.population = sorted(self.fitness_history, reverse=True)[:self.population.index(max(self.fitness_history)) + 1]

        # Refine the selected solution
        if len(self.population) > 1:
            self.population[0], self.population[1] = self.population[1], self.population[0]

        # Return the optimized parameter values and the objective function value
        return self.param_values, func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

    def mutate(self, individual):
        """
        Mutate the individual to introduce new solutions.

        Args:
            individual (tuple): The individual to mutate.

        Returns:
            tuple: The mutated individual.
        """
        mutated_individual = copy.deepcopy(individual)
        for _ in range(self.dim):
            if random.random() < 0.5:
                mutated_individual = mutated_individual[1] + np.random.normal(0, 1)
            else:
                mutated_individual = mutated_individual[0] + np.random.normal(0, 1)
        return mutated_individual

# One-line description with the main idea:
# Black box optimization using genetic algorithm with refinement to adapt to the selected solution.

# Code: