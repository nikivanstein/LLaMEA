import numpy as np
import random
import os

class EvolutionaryStrategy:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the evolutionary strategy.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0
        self.population = []
        self.fitness_history = []
        self.population_size = 100

    def __call__(self, func):
        """
        Optimize the black box function `func` using evolutionary strategy.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the population with random parameter values
        self.population = np.random.uniform(-5.0, 5.0, self.dim, size=self.population_size)

        # Accumulate noise in the objective function evaluations
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.population + self.noise * np.random.normal(0, 1, self.dim))

            # Update the population based on the accumulated noise
            self.population += self.noise * np.random.normal(0, 1, self.dim)

        # Evaluate the fitness of each individual in the population
        self.fitness_history = [func(self.population[i]) for i in range(self.population_size)]

        # Select the fittest individuals for the next generation
        self.population = self.select_fittest()

        # Return the fittest individual and its fitness
        return self.population[0], self.fitness_history[0]

    def select_fittest(self):
        """
        Select the fittest individuals for the next generation.

        Returns:
            list: A list of the fittest individuals in the population.
        """
        # Sort the population based on fitness
        self.population.sort(key=lambda x: x[1], reverse=True)

        # Select the top k individuals
        return self.population[:self.population_size // 2]

    def mutate(self, individual):
        """
        Mutate an individual with a small probability.

        Args:
            individual (list): The individual to mutate.

        Returns:
            list: The mutated individual.
        """
        # Randomly select a mutation point
        mutation_point = random.randint(0, len(individual) - 1)

        # Swap the mutation point with a random point in the individual
        mutated_individual = individual[:mutation_point] + [random.choice(individual[mutation_point:])] + individual[mutation_point + 1:]

        return mutated_individual

    def save_fitness_history(self, filename):
        """
        Save the fitness history to a file.

        Args:
            filename (str): The filename to save the fitness history.
        """
        np.save(filename, self.fitness_history)