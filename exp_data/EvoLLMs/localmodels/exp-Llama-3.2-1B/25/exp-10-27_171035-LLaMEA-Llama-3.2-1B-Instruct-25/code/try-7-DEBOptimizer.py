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

        # Initialize the population size and the number of generations
        population_size = 100
        num_generations = 100

        # Initialize the population with random solutions
        self.population = [np.random.uniform(lower_bound, upper_bound, size=(population_size, self.dim)) for _ in range(population_size)]

        # Evaluate the objective function for each individual in the population
        results = []
        for _ in range(num_generations):
            # Evaluate the objective function for each individual in the population
            fitness_values = differential_evolution(lambda x: -func(x), self.population, bounds=(lower_bound, upper_bound), x0=self.population)

            # Select the fittest individuals for the next generation
            fittest_individuals = [self.population[i] for i, _ in enumerate(results) if _ == fitness_values.x[0]]

            # Replace the least fit individuals with the fittest ones
            self.population = [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Update the population with the fittest individuals
            self.population += [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Refine the strategy based on the performance of the selected solution
            if len(self.population) > self.budget:
                # Get the best individual from the population
                best_individual = self.population[0]

                # Get the fitness of the best individual
                fitness = -func(best_individual)

                # Get the bounds of the search space
                lower_bound = -5.0
                upper_bound = 5.0

                # Refine the bounds based on the performance of the best individual
                refined_bounds = [lower_bound, upper_bound]

                # Update the bounds with the refined strategy
                self.population = [np.random.uniform(refined_bounds[0], refined_bounds[1], size=(population_size, self.dim)) for _ in range(population_size)]

                # Evaluate the objective function for each individual in the population
                results = []
                for _ in range(num_generations):
                    # Evaluate the objective function for each individual in the population
                    fitness_values = differential_evolution(lambda x: -func(x), self.population, bounds=refined_bounds, x0=self.population)

                    # Select the fittest individuals for the next generation
                    fittest_individuals = [self.population[i] for i, _ in enumerate(results) if _ == fitness_values.x[0]]

                    # Replace the least fit individuals with the fittest ones
                    self.population = [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

                    # Update the population with the fittest individuals
                    self.population += [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

                    # Refine the strategy based on the performance of the selected solution
                    if len(self.population) > self.budget:
                        # Get the best individual from the population
                        best_individual = self.population[0]

                        # Get the fitness of the best individual
                        fitness = -func(best_individual)

                        # Get the bounds of the search space
                        lower_bound = -5.0
                        upper_bound = 5.0

                        # Refine the bounds based on the performance of the best individual
                        refined_bounds = [lower_bound, upper_bound]

                        # Update the bounds with the refined strategy
                        self.population = [np.random.uniform(refined_bounds[0], refined_bounds[1], size=(population_size, self.dim)) for _ in range(population_size)]

                # Return the optimized function and its value
                return func(best_individual), fitness

    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual in the population.

        Args:
            individual (numpy array): The individual to evaluate.

        Returns:
            float: The fitness of the individual.
        """
        # Get the bounds of the search space
        lower_bound = -5.0
        upper_bound = 5.0

        # Evaluate the objective function for the individual
        fitness = -func(individual)

        # Refine the bounds based on the performance of the individual
        refined_bounds = [lower_bound, upper_bound]

        # Update the bounds with the refined strategy
        individual = np.random.uniform(refined_bounds[0], refined_bounds[1], size=(1, self.dim))

        # Evaluate the objective function for the individual
        fitness = -func(individual)

        return fitness