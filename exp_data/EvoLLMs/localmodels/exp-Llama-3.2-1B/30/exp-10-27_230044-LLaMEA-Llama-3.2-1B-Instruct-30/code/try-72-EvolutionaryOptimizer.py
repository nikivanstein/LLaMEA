import numpy as np
import random
import math

class EvolutionaryOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the evolutionary optimizer with a given budget and dimension.

        Args:
            budget (int): The maximum number of function evaluations.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.population = None
        self.fitnesses = None

    def __call__(self, func):
        """
        Optimize the black box function using evolutionary optimization.

        Args:
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized value, fitness, and population.
        """
        # Initialize the population with random individuals
        self.population = [np.random.rand(self.dim) for _ in range(100)]

        # Define the mutation and selection operators
        def mutate(individual):
            # Randomly change a single element in the individual
            mutated_individual = individual.copy()
            mutated_individual[random.randint(0, self.dim-1)] += random.uniform(-1, 1)
            return mutated_individual

        def select(population):
            # Select the fittest individuals for the next generation
            fitnesses = [individual fitness for individual, fitness in zip(population, self.fitnesses)]
            return np.array(population)[np.argsort(fitnesses)[-self.budget:]]

        # Run the evolutionary algorithm
        for _ in range(self.budget):
            # Generate a new population
            new_population = [mutate(individual) for individual in self.population]

            # Evaluate the fitness of the new population
            fitnesses = [individual fitness for individual, fitness in zip(new_population, self.fitnesses)]
            new_population = select(new_population)

            # Replace the old population with the new one
            self.population = new_population

        # Return the best individual and its fitness
        return self.population[0], np.max(fitnesses), self.population

    def evaluateBBOB(self, func):
        """
        Evaluate the black box function using the BBOB test suite.

        Args:
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized value, fitness, and population.
        """
        # Initialize the population with random individuals
        self.population = [np.random.rand(self.dim) for _ in range(100)]

        # Define the mutation and selection operators
        def mutate(individual):
            # Randomly change a single element in the individual
            mutated_individual = individual.copy()
            mutated_individual[random.randint(0, self.dim-1)] += random.uniform(-1, 1)
            return mutated_individual

        def select(population):
            # Select the fittest individuals for the next generation
            fitnesses = [individual fitness for individual, fitness in zip(population, np.random.rand(100, 24))]

            # Return the top 24 individuals with the highest fitness
            return np.array(population)[np.argsort(fitnesses)[:24]]

        # Run the evolutionary algorithm
        for _ in range(100):
            # Generate a new population
            new_population = [mutate(individual) for individual in self.population]

            # Evaluate the fitness of the new population
            fitnesses = [individual fitness for individual, fitness in zip(new_population, np.random.rand(100, 24))]

            # Replace the old population with the new one
            self.population = select(new_population)

        # Return the best individual and its fitness
        return self.population[0], np.max(fitnesses), self.population

# Description: Evolutionary Algorithm with Adaptive Line Search for Black Box Optimization
# Code: 