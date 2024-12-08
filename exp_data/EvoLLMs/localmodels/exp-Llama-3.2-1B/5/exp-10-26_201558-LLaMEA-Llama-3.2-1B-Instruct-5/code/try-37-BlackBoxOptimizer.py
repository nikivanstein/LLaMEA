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

            # Replace the old population with the new population
            population = new_population

            # Apply the mutation strategy
            mutated_population = np.copy(population)
            for _ in range(population_size // 2):
                mutated_individual = mutated_population[random.randint(0, population_size - 1)]
                mutated_individual = self.evaluate_fitness(mutated_individual, mutated_individual, mutated_individual, mutated_individual)
                mutated_population[random.randint(0, population_size - 1)] = mutated_individual

        # Return the optimized parameters and the optimized function value
        return population, func(population)

    def evaluate_fitness(self, individual, fitness, individual_fitness, individual_new):
        """
        Evaluate the fitness of the individual and apply the mutation strategy.

        Parameters:
        individual (int): The individual to evaluate.
        fitness (float): The fitness value of the individual.
        individual_fitness (float): The fitness value of the individual's fitness function.
        individual_new (float): The fitness value of the individual's new fitness function.

        Returns:
        float: The fitness value of the individual after applying the mutation strategy.
        """
        # Apply the mutation strategy
        mutated_individual = individual
        mutated_individual_fitness = individual_fitness
        mutated_individual_new = individual_new

        # Apply the mutation strategy
        if random.random() < 0.05:
            mutated_individual = self.evaluate_fitness(mutated_individual, mutated_individual_fitness, mutated_individual_new, mutated_individual_new)
            mutated_individual_fitness = mutated_individual_fitness
            mutated_individual_new = mutated_individual_new

        # Return the fitness value of the individual after applying the mutation strategy
        return mutated_individual_fitness