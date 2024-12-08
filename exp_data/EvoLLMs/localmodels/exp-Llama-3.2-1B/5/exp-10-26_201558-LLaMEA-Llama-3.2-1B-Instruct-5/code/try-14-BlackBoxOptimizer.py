# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
import numpy as np
import random

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
        self.population_history = []

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
            self.population_history.append((population, func(population)))

            # Refine the strategy with a novel mutation
            if random.random() < 0.05:
                # Select a random individual and mutate it
                mutated_individual = population[np.random.randint(0, self.population_size)]
                mutated_individual = np.clip(mutated_individual, -5.0, 5.0)
                mutated_individual = np.clip(mutated_individual, -5.0, 5.0)
                mutated_individual = np.random.uniform(-5.0, 5.0, mutated_individual.shape)
                mutated_individual = mutated_individual / np.linalg.norm(mutated_individual)

                # Replace the old population with the mutated population
                population = np.concatenate([population[:fittest_individuals.size // 2], mutated_individual])

        # Return the optimized parameters and the optimized function value
        return population, func(population)

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
black_box_optimizer = BlackBoxOptimizer(budget=1000, dim=10)