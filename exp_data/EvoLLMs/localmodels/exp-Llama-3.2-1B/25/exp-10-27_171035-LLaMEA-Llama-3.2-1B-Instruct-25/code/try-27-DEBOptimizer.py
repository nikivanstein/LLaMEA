import numpy as np
from scipy.optimize import differential_evolution
import random

class DEBOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the DEBOptimizer with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.func = None
        self.population = None
        self.population_size = 100
        self.num_generations = 100
        self.fitness_values = []
        self.adaptive_mutations = 0.0

    def __call__(self, func):
        """
        Optimize a black box function using DEBOptimizer.

        Args:
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized function and its value.
        """
        # Get the bounds of the search space
        lower_bound = -5.0
        upper_bound = 5.0

        # Initialize the population with random solutions
        self.population = [np.random.uniform(lower_bound, upper_bound, size=(self.population_size, self.dim)) for _ in range(self.population_size)]

        # Evaluate the objective function for each individual in the population
        results = []
        for _ in range(self.num_generations):
            # Evaluate the objective function for each individual in the population
            fitness_values = differential_evolution(lambda x: -func(x), self.population, bounds=(lower_bound, upper_bound), x0=self.population)

            # Select the fittest individuals for the next generation
            fittest_individuals = [self.population[i] for i, _ in enumerate(results) if _ == fitness_values.x[0]]

            # Replace the least fit individuals with the fittest ones
            self.population = [self.population[i] for i in range(self.population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Update the population with the fittest individuals
            self.population += [self.population[i] for i in range(self.population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Check if the population has reached the budget
            if len(self.population) > self.budget:
                break

        # Return the optimized function and its value
        return func(self.population[0]), -func(self.population[0])

    def mutate(self, individual):
        """
        Apply adaptive mutation to the selected solution.

        Args:
            individual (numpy array): A single individual in the population.
        """
        # Select the fittest individual
        fittest_individual = self.population[self.fitness_values.index(max(self.fitness_values))]

        # Calculate the mutation probability
        mutation_probability = self.adaptive_mutations

        # Apply adaptive mutation
        if random.random() < mutation_probability:
            # Randomly select a new individual from the search space
            new_individual = np.random.uniform(lower_bound, upper_bound, size=self.dim)

            # Calculate the fitness of the new individual
            new_fitness = -func(new_individual)

            # Update the individual and its fitness value
            self.population[self.fitness_values.index(new_individual)] = new_individual
            self.fitness_values.append(new_fitness)

        return individual, new_individual

# One-line description with the main idea
# Evolutionary Black Box Optimization using Differential Evolution with Adaptive Mutation
# This algorithm optimizes a black box function using differential evolution with adaptive mutation, which allows for the selection of fittest individuals based on their fitness values and the application of adaptive mutation to refine the strategy.

# Code: 