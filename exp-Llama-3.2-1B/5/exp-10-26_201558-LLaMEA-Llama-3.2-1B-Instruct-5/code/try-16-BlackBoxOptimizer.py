# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
import numpy as np
import random
from collections import deque
import time

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
        self.population_size = 100
        self.population_history = deque(maxlen=100)
        self mutation_rate = 0.05
        self.budget_history = deque(maxlen=100)
        self.best_individual = None
        self.best_function_value = -np.inf

    def __call__(self, func):
        """
        Optimize the black box function `func` using the given budget and search space.

        Parameters:
        func (function): The black box function to optimize.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population with random parameters
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

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

            # Update the population history
            self.population_history.append(population)

            # Update the budget history
            self.budget_history.append(len(func_values))

            # Update the best individual and function value
            if len(func_values) > len(self.best_function_value):
                self.best_individual = population
                self.best_function_value = func_values[-1]

        # Return the optimized parameters and the optimized function value
        return population, func(population)

    def mutate(self, individual):
        """
        Mutate the individual by changing a random element.

        Parameters:
        individual (numpy array): The individual to mutate.

        Returns:
        numpy array: The mutated individual.
        """
        # Randomly select an element to mutate
        idx = random.randint(0, self.dim - 1)

        # Mutate the element
        individual[idx] = random.uniform(-5.0, 5.0)

        # Return the mutated individual
        return individual

    def select_parents(self, population):
        """
        Select parents for the next generation based on the fitness values.

        Parameters:
        population (numpy array): The population to select parents from.

        Returns:
        tuple: The selected parents.
        """
        # Calculate the fitness values
        fitness_values = np.array([func(population[i]) for i in range(len(population))])

        # Select the fittest parents
        fittest_parents = np.argsort(fitness_values)[::-1][:self.population_size // 2]

        # Return the selected parents
        return fittest_parents