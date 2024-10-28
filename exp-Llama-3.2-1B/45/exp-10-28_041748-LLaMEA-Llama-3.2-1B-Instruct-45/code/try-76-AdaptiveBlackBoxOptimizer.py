import random
import numpy as np
import logging
from aoc_logger import aoc_logger

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim, mutation_rate=0.1, crossover_rate=0.5):
        """
        Initialize the Adaptive Black Box Optimization algorithm.

        Args:
        - budget (int): The maximum number of function evaluations.
        - dim (int): The dimensionality of the search space.
        - mutation_rate (float, optional): The rate of mutation. Defaults to 0.1.
        - crossover_rate (float, optional): The rate of crossover. Defaults to 0.5.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.logger = aoc_logger()

    def __call__(self, func):
        """
        Evaluate the given black box function and select the top-performing individuals.

        Args:
        - func (function): The black box function to optimize.

        Returns:
        - new_population (list): The new population of individuals after optimization.
        """
        # Evaluate the function with the given budget
        func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
        
        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-self.population_size:]
        
        # Create a new population by crossover and mutation
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            new_population.append(child)
        
        # Replace the old population with the new one
        self.population = new_population
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Return the best individual
        best_individual = np.argmax(new_func_evaluations)
        return new_population[best_individual]

    def optimize(self, func, budget):
        """
        Optimize the given black box function using the Adaptive Black Box Optimization algorithm.

        Args:
        - func (function): The black box function to optimize.
        - budget (int): The maximum number of function evaluations.

        Returns:
        - best_individual (int): The index of the best individual.
        """
        # Initialize the population
        self.population = []
        
        # Evaluate the function with the given budget
        func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
        
        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-self.population_size:]
        
        # Create a new population by crossover and mutation
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            self.population.append(child)
        
        # Replace the old population with the new one
        self.population = self.population[:budget]
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(self.population))])
        
        # Return the best individual
        best_individual = np.argmax(new_func_evaluations)
        return best_individual

# One-Liner Description: 
# Adaptive Black Box Optimization using Genetic Algorithms with Tunable Mutation Rates and Crossover Rates.

# Code: