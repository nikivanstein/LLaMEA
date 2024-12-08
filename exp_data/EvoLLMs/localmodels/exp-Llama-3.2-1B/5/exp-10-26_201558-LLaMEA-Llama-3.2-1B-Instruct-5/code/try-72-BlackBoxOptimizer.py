# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
import numpy as np
import random
import operator
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a given budget and dimensionality.

        Parameters:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimize the black box function `func` using the given budget and search space.

        Parameters:
        func (function): The black box function to optimize.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population size
        population_size = 100

        # Initialize the population with random parameters
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))

        # Evaluate the function for each individual in the population
        for _ in range(self.budget):
            # Evaluate the function for each individual in the population
            func_values = func(population)

            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func_values)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            population = new_population

        # Define the mutation strategy
        def mutate(individual, mutation_rate):
            # Randomly select an index to mutate
            index = random.randint(0, self.dim - 1)

            # Apply the mutation
            individual[index] += random.uniform(-1, 1)

            # Return the mutated individual
            return individual

        # Define the crossover strategy
        def crossover(parent1, parent2):
            # Select a random crossover point
            crossover_point = random.randint(0, self.dim - 1)

            # Combine the two parents
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

            # Return the child
            return child

        # Define the selection strategy
        def selection(population):
            # Select the fittest individuals based on the fitness values
            fitness_values = np.array([func(individual) for individual in population])
            fittest_individuals = np.argsort(fitness_values)[::-1][:self.population_size // 2]

            # Return the fittest individuals
            return fittest_individuals

        # Initialize the population with random parameters
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))

        # Evaluate the function for each individual in the population
        for _ in range(self.budget):
            # Evaluate the function for each individual in the population
            func_values = func(population)

            # Select the fittest individuals based on the function values
            fittest_individuals = selection(population)

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            population = new_population

            # Apply the mutation strategy
            population = np.vectorize(mutate)(population, mutation_rate=0.05)

            # Apply the crossover strategy
            population = np.vectorize(crossover)(population, p1=population[:fittest_individuals.size // 2], p2=population[fittest_individuals.size // 2:])

        # Return the optimized parameters and the optimized function value
        return population, func(population)

# One-line description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
# BlackBoxOptimizer: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# ```python