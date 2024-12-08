import numpy as np
from scipy.optimize import minimize
from collections import deque
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

        # Define the mutation function
        def mutation(individual, mutation_rate):
            if random.random() < mutation_rate:
                # Select a random dimension and mutate the value
                dim_idx = random.randint(0, self.dim - 1)
                individual[dim_idx] += random.uniform(-1, 1)
            return individual

        # Define the crossover function
        def crossover(parent1, parent2):
            # Select a random crossover point
            crossover_point = random.randint(0, self.dim - 1)

            # Create a new offspring by combining the two parents
            offspring = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])

            # Replace the old parents with the new offspring
            parent1 = np.concatenate([parent1, offspring])
            parent2 = np.concatenate([parent2, offspring])

            return parent1, parent2

        # Define the selection function
        def selection(population, fitness):
            # Select the fittest individuals based on their fitness
            return np.argsort(fitness)[::-1][:self.population_size // 2]

        # Define the tournament selection function
        def tournament_selection(population, k):
            # Select k individuals from the population
            tournament = np.random.choice(population, size=k, replace=False)

            # Evaluate the fitness of each individual in the tournament
            fitness = np.array([func(individual) for individual in tournament])

            # Select the fittest individual based on the fitness
            return np.argsort(fitness)[::-1][:k]

        # Define the fitness function
        def fitness(individual):
            # Evaluate the function for the individual
            func_value = func(individual)
            return func_value

        # Define the bounds for the individual parameters
        bounds = np.array([[-5.0, 5.0] for _ in range(self.dim)])

        # Initialize the best individual and its fitness
        best_individual = None
        best_fitness = -np.inf

        # Initialize the queue for the evolution
        queue = deque()

        # Iterate through the population
        while len(queue) > 0 and fitness(best_individual) > best_fitness:
            # Get the best individual and its fitness
            best_individual, best_fitness = queue.popleft()

            # Create a new population by combining the best individual with a random mutation
            new_individual = mutation(best_individual, 0.05)

            # Replace the old population with the new population
            queue.append((new_individual, fitness(new_individual)))

            # Add the new individual to the queue
            queue.append((new_individual, fitness(new_individual)))

        # Return the best individual and its fitness
        return best_individual, best_fitness

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy

# Code: