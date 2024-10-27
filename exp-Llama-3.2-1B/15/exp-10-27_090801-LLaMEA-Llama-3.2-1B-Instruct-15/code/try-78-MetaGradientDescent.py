import numpy as np
import random

class MetaGradientDescent:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-gradient descent algorithm.

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
        Optimize the black box function `func` using meta-gradient descent.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the parameter values to random values within the search space
        self.param_values = np.random.uniform(-5.0, 5.0, self.dim)

        # Accumulate noise in the objective function evaluations
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise
            self.param_values += self.noise * np.random.normal(0, 1, self.dim)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

class MetaGeneticAlgorithm:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-genetic algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0

    def __call__(self, func, population_size, mutation_rate):
        """
        Optimize the black box function `func` using the meta-genetic algorithm.

        Args:
            func (callable): The black box function to optimize.
            population_size (int): The size of the population.
            mutation_rate (float): The mutation rate.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the population with random parameter values
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(population_size)]

        # Evaluate the fitness of each individual in the population
        fitnesses = [self.evaluate_fitness(individual, func) for individual in population]

        # Select the fittest individuals
        selected_individuals = np.argsort(fitnesses)[:int(population_size/2)]

        # Create a new population by mutating the selected individuals
        new_population = []
        for _ in range(population_size):
            # Select a random individual from the selected individuals
            individual = selected_individuals[np.random.randint(0, len(selected_individuals))]

            # Evaluate the fitness of the individual
            fitness = self.evaluate_fitness(individual, func)

            # Mutate the individual with the specified mutation rate
            mutated_individual = individual + self.noise * np.random.normal(0, 1, self.dim)

            # Add the mutated individual to the new population
            new_population.append(mutated_individual)

        # Evaluate the fitness of the new population
        fitnesses = [self.evaluate_fitness(individual, func) for individual in new_population]

        # Select the fittest individuals from the new population
        selected_individuals = np.argsort(fitnesses)[:int(population_size/2)]

        # Return the optimized parameter values and the objective function value
        return selected_individuals, fitnesses[int(population_size/2)], fitnesses[int(population_size/2)].max()

    def evaluate_fitness(self, individual, func):
        """
        Evaluate the fitness of the individual `individual` using the function `func`.

        Args:
            individual (numpy array): The individual to evaluate.
            func (callable): The function to evaluate the individual with.

        Returns:
            float: The fitness of the individual.
        """
        # Evaluate the objective function with the accumulated noise
        func_value = func(individual + self.noise * np.random.normal(0, 1, self.dim))

        # Return the fitness of the individual
        return func_value

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 