import numpy as np
import random

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the evolutionary algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.population = 100
        self.population_size = 100
        self.population_refinement = 0.1

    def __call__(self, func):
        """
        Optimize the black box function `func` using evolutionary algorithm.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the population with random parameter values
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

        # Evaluate the objective function with accumulated noise
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.population[-1] + self.noise_level * np.random.normal(0, 1, self.dim))

            # Update the population based on the accumulated noise
            self.population = [p + self.noise_level * np.random.normal(0, 1, self.dim) for p in self.population]

        # Refine the population based on the fitness of the best individual
        best_individual = self.population[0]
        best_fitness = self.evaluate_fitness(best_individual)
        for _ in range(self.population_size):
            fitness = self.evaluate_fitness(self.population[_])
            if fitness > best_fitness:
                best_individual = self.population[_]
                best_fitness = fitness

        # Return the optimized parameter values and the objective function value
        return best_individual, func(best_individual)

    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual.

        Args:
            individual (numpy.ndarray): The individual to evaluate.

        Returns:
            float: The fitness of the individual.
        """
        # Evaluate the objective function
        func_value = func(individual)

        # Refine the fitness based on the probability
        fitness = func_value + self.population_refinement * np.random.normal(0, 1, self.dim)

        return fitness

# One-line description with the main idea
# Evolutionary Algorithm with Adaptive Population Refinement