import numpy as np
import random
import pickle

class EvolutionaryStrategy:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the evolutionary strategy algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0

    def __call__(self, func):
        """
        Optimize the black box function `func` using evolutionary strategies.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the population of individuals
        population = self.generate_population(self.budget)

        # Iterate until convergence or maximum iterations reached
        while True:
            # Evaluate the fitness of each individual in the population
            fitness = [self.evaluate_fitness(individual, func) for individual in population]

            # Select the fittest individuals for the next generation
            fittest_individuals = self.select_fittest(population, fitness)

            # Create a new generation of individuals
            new_population = self.generate_new_population(fittest_individuals, func, self.dim)

            # Update the population for the next iteration
            population = new_population

            # Check for convergence
            if np.all(population == fittest_individuals):
                break

            # Update the noise level for the next iteration
            self.noise_level *= 0.8

        # Return the optimized parameter values and the objective function value
        return self.evaluate_fitness(fittest_individuals[0], func), fitness[0]

    def generate_population(self, budget):
        """
        Generate a population of individuals using evolutionary strategies.

        Args:
            budget (int): The maximum number of function evaluations allowed.

        Returns:
            list: A list of individuals in the population.
        """
        population = []
        for _ in range(budget):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def select_fittest(self, population, fitness):
        """
        Select the fittest individuals for the next generation.

        Args:
            population (list): A list of individuals in the population.
            fitness (list): A list of fitness values corresponding to the individuals.

        Returns:
            list: A list of fittest individuals.
        """
        fittest_individuals = []
        for i in range(len(population)):
            fittest_individuals.append(population[i])
            if fitness[i] == max(fitness):
                fittest_individuals.append(population[i])
        return fittest_individuals

    def generate_new_population(self, fittest_individuals, func, dim):
        """
        Generate a new population of individuals using evolutionary strategies.

        Args:
            fittest_individuals (list): A list of fittest individuals.
            func (callable): The black box function to optimize.
            dim (int): The dimensionality of the problem.

        Returns:
            list: A list of new individuals in the population.
        """
        new_population = []
        for _ in range(len(fittest_individuals)):
            individual = np.random.uniform(-5.0, 5.0, dim)
            while True:
                fitness_value = func(individual)
                if np.all(individual == fittest_individuals):
                    break
                individual += np.random.normal(0, 1, dim)
            new_population.append(individual)
        return new_population

    def evaluate_fitness(self, individual, func):
        """
        Evaluate the fitness of an individual using the given function.

        Args:
            individual (numpy.ndarray): An individual in the population.
            func (callable): The black box function to optimize.

        Returns:
            float: The fitness value of the individual.
        """
        return func(individual)

# One-Liner Description: Novel Metaheuristic Algorithm for Black Box Optimization using Evolutionary Strategies
# Code: 