import random
import numpy as np
import math
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Generate a random point in the search space
            point = self.search_space[np.random.randint(0, self.dim)]

            # Evaluate the function at the current point
            value = func(point)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > best_value:
                best_value = value
                best_index = point

        # Return the optimized value
        return best_value

class GeneticAlgorithm(BlackBoxOptimizer):
    def __init__(self, budget, dim):
        """
        Initialize the GeneticAlgorithm with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        super().__init__(budget, dim)

    def __call__(self, func):
        """
        Optimize the black box function using the GeneticAlgorithm.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the population size and the number of generations
        population_size = 100
        num_generations = 100

        # Initialize the population with random individuals
        population = np.random.uniform(self.search_space, size=(population_size, self.dim))

        # Evaluate the fitness of each individual in the population
        fitnesses = np.array([self(func(individual)) for individual in population])

        # Select the fittest individuals to reproduce
        fittest_indices = np.argsort(fitnesses)[-self.budget:]

        # Create a new generation by crossover and mutation
        new_population = np.zeros((population_size, self.dim))
        for i in range(population_size):
            parent1, parent2 = random.sample(fittest_indices, 2)
            child = (parent1 + parent2) / 2
            new_population[i] = np.random.uniform(self.search_space, size=(dim,))
            for j in range(dim):
                if random.random() < 0.5:
                    new_population[i, j] = (child[j] + new_population[i, j]) / 2
                else:
                    new_population[i, j] = (child[j] + random.uniform(-1, 1)) / 2

        # Evaluate the fitness of the new population
        new_fitnesses = np.array([self.func(individual) for individual in new_population])

        # Select the fittest individuals to reproduce
        fittest_indices = np.argsort(new_fitnesses)[-self.budget:]

        # Create a new generation by crossover and mutation
        new_population = np.zeros((population_size, self.dim))
        for i in range(population_size):
            parent1, parent2 = random.sample(fittest_indices, 2)
            child = (parent1 + parent2) / 2
            new_population[i] = np.random.uniform(self.search_space, size=(dim,))
            for j in range(dim):
                if random.random() < 0.5:
                    new_population[i, j] = (child[j] + new_population[i, j]) / 2
                else:
                    new_population[i, j] = (child[j] + random.uniform(-1, 1)) / 2

        # Return the fittest individual in the new population
        return np.min(new_fitnesses)

# Description: Novel Metaheuristic Algorithm for Black Box Optimization using Evolutionary Strategies
# Code: 