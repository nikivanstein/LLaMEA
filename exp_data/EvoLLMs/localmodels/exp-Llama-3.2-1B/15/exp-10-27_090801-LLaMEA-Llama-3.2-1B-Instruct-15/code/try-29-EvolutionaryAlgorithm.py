import numpy as np
import random
import os

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
        self.noise = 0

    def __call__(self, func, iterations=1000, mutation_rate=0.01):
        """
        Optimize the black box function `func` using evolutionary algorithms.

        Args:
            func (callable): The black box function to optimize.
            iterations (int, optional): The number of iterations. Defaults to 1000.
            mutation_rate (float, optional): The mutation rate. Defaults to 0.01.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the population size
        population_size = 100

        # Initialize the population with random parameter values
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(population_size)]

        # Run the evolutionary algorithm
        for _ in range(iterations):
            # Evaluate the fitness of each individual in the population
            fitness = [self.evaluate_fitness(individual, func) for individual in population]

            # Select the fittest individuals
            fittest_individuals = np.argsort(fitness)[-population_size:]

            # Mutate the fittest individuals
            for _ in range(int(population_size * mutation_rate)):
                fittest_individuals = [individuals[fittest_individuals.index(individual)] for individual in population]

            # Replace the fittest individuals with new ones
            population = [individuals[fittest_individuals.index(individual)] for individual in population]

            # Update the noise level
            self.noise += 0.1

            # Evaluate the fitness of each individual in the population
            fitness = [self.evaluate_fitness(individual, func) for individual in population]

            # Select the fittest individuals
            fittest_individuals = np.argsort(fitness)[-population_size:]

            # Mutate the fittest individuals
            for _ in range(int(population_size * mutation_rate)):
                fittest_individuals = [individuals[fittest_individuals.index(individual)] for individual in population]

            # Replace the fittest individuals with new ones
            population = [individuals[fittest_individuals.index(individual)] for individual in population]

        # Return the optimized parameter values and the objective function value
        return population[0], fitness[0]

    def evaluate_fitness(self, individual, func):
        """
        Evaluate the fitness of an individual.

        Args:
            individual (numpy.ndarray): The individual to evaluate.
            func (callable): The black box function to evaluate.

        Returns:
            float: The fitness of the individual.
        """
        # Evaluate the objective function with accumulated noise
        return func(individual + self.noise * np.random.normal(0, 1, self.dim))

# Example usage
def test_func(x):
    return np.sum(x ** 2)

algorithm = EvolutionaryAlgorithm(budget=100, dim=10, noise_level=0.1)

optimized_individual, optimized_fitness = algorithm(__call__(test_func, iterations=1000))

print(f"Optimized individual: {optimized_individual}")
print(f"Optimized fitness: {optimized_fitness}")