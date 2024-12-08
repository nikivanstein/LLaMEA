import numpy as np
import random
import os

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim, noise_level=0.1, mutation_rate=0.1, exploration_rate=0.15):
        """
        Initialize the adaptive black box optimizer.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
            mutation_rate (float, optional): The rate of mutation. Defaults to 0.1.
            exploration_rate (float, optional): The rate of exploration. Defaults to 0.15.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.mutation_rate = mutation_rate
        self.exploration_rate = exploration_rate
        self.noise = 0
        self.population = None
        self.best_individual = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        """
        Optimize the black box function `func` using adaptive black box optimization.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the population with random parameter values
        self.population = np.random.uniform(-5.0, 5.0, self.dim) + self.noise * np.random.normal(0, 1, self.dim)

        # Initialize the best individual and its fitness
        self.best_individual = self.population
        self.best_fitness = func(self.population)

        # Initialize the counter for function evaluations
        self.evaluations = 0

        # Evaluate the function until the budget is reached or the best individual is found
        while self.evaluations < self.budget:
            # Evaluate the function with the current population
            func_value = func(self.population)

            # Update the population based on the exploration rate
            if np.random.rand() < self.exploration_rate:
                # Perform mutation on a random individual
                mutated_individual = self.population + self.noise * np.random.normal(0, 1, self.dim)
                mutated_individual = np.clip(mutated_individual, -5.0, 5.0)
                mutated_individual = np.clip(mutated_individual, -5.0, 5.0)

                # Update the population
                self.population = np.concatenate((self.population, mutated_individual))

            # Update the best individual and its fitness
            if func_value < self.best_fitness:
                self.best_individual = self.population
                self.best_fitness = func_value

            # Increment the evaluation counter
            self.evaluations += 1

        # Return the optimized parameter values and the objective function value
        return self.best_individual, self.best_fitness

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolutionary Strategies
# This algorithm optimizes the black box function using an evolutionary strategy, where the population is updated based on exploration and mutation, and the best individual is selected based on a probability distribution.