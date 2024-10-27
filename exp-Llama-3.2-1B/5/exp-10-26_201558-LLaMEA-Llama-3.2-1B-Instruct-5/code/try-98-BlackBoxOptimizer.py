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

            # Apply mutation to the new population
            mutated_population = self.mutate(new_population)

            # Replace the old population with the new population
            population = mutated_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)

    def mutate(self, population):
        """
        Apply a novel mutation strategy to the given population.

        Parameters:
        population (numpy array): The population to mutate.

        Returns:
        numpy array: The mutated population.
        """
        # Define the mutation operators
        def crossover(parent1, parent2):
            # Select a random crossover point
            crossover_point = random.randint(1, self.dim)

            # Create a new offspring by combining the two parents
            offspring = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

            # Return the offspring
            return offspring

        def mutation(individual):
            # Select a random mutation point
            mutation_point = random.randint(1, self.dim)

            # Apply the mutation
            mutated_individual = individual[:mutation_point] + [random.uniform(-5.0, 5.0)] + individual[mutation_point:]

            # Return the mutated individual
            return mutated_individual

        # Apply crossover and mutation to each individual in the population
        mutated_population = np.array([crossover(population[i], population[i]) for i in range(population.shape[0])])

        # Apply mutation to each individual in the mutated population
        mutated_population = np.array([mutation(individual) for individual in mutated_population])

        # Return the mutated population
        return mutated_population

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Optimizes the black box function using a population-based approach with mutation to refine the strategy