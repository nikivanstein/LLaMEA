import numpy as np
import random

class GeneticBlackBoxOptimizer:
    """
    A genetic algorithm that optimizes black box functions by leveraging the power of evolutionary strategies.

    Attributes:
    ----------
    budget : int
        The maximum number of function evaluations allowed.
    dim : int
        The dimensionality of the search space.
    population_size : int
        The population size for the genetic algorithm.
    mutation_rate : float
        The rate at which the genetic algorithm mutates the population.
    bounds : list
        The bounds for the search space.
    fitness_function : function
        The function to evaluate the fitness of an individual.

    Methods:
    -------
    __init__(self, budget, dim, population_size, mutation_rate, bounds, fitness_function)
        Initializes the genetic algorithm with the given parameters.
    def __call__(self, func)
        Optimizes the black box function `func` using `self.budget` function evaluations.
    def select_parents(self, population, budget)
        Selects the parents for the next generation based on the fitness scores.
    def mutate_parents(self, parents, mutation_rate)
        Mutates the parents based on the mutation rate.
    def crossover(self, parents)
        Crossover the parents to generate the offspring.
    def evolve(self, population, budget)
        Evolves the population using the genetic algorithm.
    """

    def __init__(self, budget, dim, population_size, mutation_rate, bounds, fitness_function):
        """
        Initializes the genetic algorithm with the given parameters.

        Parameters:
        ----------
        budget : int
            The maximum number of function evaluations allowed.
        dim : int
            The dimensionality of the search space.
        population_size : int
            The population size for the genetic algorithm.
        mutation_rate : float
            The rate at which the genetic algorithm mutates the population.
        bounds : list
            The bounds for the search space.
        fitness_function : function
            The function to evaluate the fitness of an individual.
        """
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.bounds = bounds
        self.fitness_function = fitness_function

    def __call__(self, func):
        """
        Optimizes the black box function `func` using `self.budget` function evaluations.

        Parameters:
        ----------
        func : function
            The black box function to optimize.

        Returns:
        -------
        tuple
            A tuple containing the optimized parameters and the optimized function value.
        """
        # Define the bounds for the search space
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]

        # Initialize the population with random parameters
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        # Evaluate the fitness of each individual
        fitness = [self.fitness_function(individual) for individual in population]

        # Select the parents based on the fitness scores
        parents = self.select_parents(population, self.budget)

        # Mutate the parents based on the mutation rate
        mutated_parents = self.mutate_parents(parents, self.mutation_rate)

        # Crossover the parents to generate the offspring
        offspring = self.crossover(mutated_parents)

        # Evolve the population using the genetic algorithm
        self.evolve(offspring, self.budget)

        # Return the optimized parameters and the optimized function value
        return offspring[0], self.fitness_function(offspring[0])

    def select_parents(self, population, budget):
        """
        Selects the parents for the next generation based on the fitness scores.

        Parameters:
        ----------
        population : numpy.ndarray
            The population of individuals.
        budget : int
            The maximum number of function evaluations allowed.

        Returns:
        -------
        numpy.ndarray
            The selected parents for the next generation.
        """
        # Evaluate the fitness of each individual
        fitness = np.array([self.fitness_function(individual) for individual in population])

        # Select the parents based on the fitness scores
        indices = np.argsort(fitness)[:budget]
        parents = population[indices]

        # Return the selected parents
        return parents

    def mutate_parents(self, parents, mutation_rate):
        """
        Mutates the parents based on the mutation rate.

        Parameters:
        ----------
        parents : numpy.ndarray
            The parents for the next generation.
        mutation_rate : float
            The rate at which the parents are mutated.

        Returns:
        -------
        numpy.ndarray
            The mutated parents.
        """
        # Evaluate the fitness of each individual
        fitness = np.array([self.fitness_function(individual) for individual in parents])

        # Mutate the parents based on the mutation rate
        mutated_parents = parents.copy()
        for i in range(parents.shape[0]):
            if np.random.rand() < mutation_rate:
                mutated_parents[i] = np.random.uniform(-5.0, 5.0, self.dim)

        # Return the mutated parents
        return mutated_parents

    def crossover(self, parents):
        """
        Crossover the parents to generate the offspring.

        Parameters:
        ----------
        parents : numpy.ndarray
            The parents for the next generation.

        Returns:
        -------
        numpy.ndarray
            The offspring.
        """
        # Evaluate the fitness of each individual
        fitness = np.array([self.fitness_function(individual) for individual in parents])

        # Crossover the parents to generate the offspring
        offspring = np.array([parents[i] for i in np.argsort(fitness)[:self.population_size]])
        return offspring

    def evolve(self, population, budget):
        """
        Evolves the population using the genetic algorithm.

        Parameters:
        ----------
        population : numpy.ndarray
            The population of individuals.
        budget : int
            The maximum number of function evaluations allowed.
        """
        # Evaluate the fitness of each individual
        fitness = np.array([self.fitness_function(individual) for individual in population])

        # Evolve the population based on the fitness scores
        for _ in range(self.budget):
            # Select the parents based on the fitness scores
            parents = self.select_parents(population, self.budget)

            # Mutate the parents based on the mutation rate
            mutated_parents = self.mutate_parents(parents, self.mutation_rate)

            # Crossover the parents to generate the offspring
            offspring = self.crossover(mutated_parents)

            # Evaluate the fitness of the offspring
            fitness = np.array([self.fitness_function(individual) for individual in offspring])

            # Replace the old population with the new population
            population = offspring


# One-line description with the main idea
# "GeneticBlackBoxOptimizer" is a novel genetic algorithm that optimizes black box functions by leveraging the power of evolutionary strategies."

# Code: