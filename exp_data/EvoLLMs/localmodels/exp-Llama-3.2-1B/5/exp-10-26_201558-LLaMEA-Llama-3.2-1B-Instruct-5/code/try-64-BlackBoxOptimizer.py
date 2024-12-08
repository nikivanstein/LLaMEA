# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
import numpy as np
import random
import operator

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
        def mutate(individual):
            # Generate a random mutation vector
            mutation_vector = np.random.uniform(-1.0, 1.0, self.dim)

            # Apply the mutation to the individual
            mutated_individual = individual + mutation_vector

            # Clip the mutated individual to the search space
            mutated_individual = np.clip(mutated_individual, -5.0, 5.0)

            return mutated_individual

        # Define the crossover function
        def crossover(parent1, parent2):
            # Select a random crossover point
            crossover_point = random.randint(1, self.dim)

            # Split the parents into two segments
            segment1 = parent1[:crossover_point]
            segment2 = parent2[crossover_point:]

            # Combine the two segments
            child = np.concatenate([segment1, segment2])

            return child

        # Define the selection function
        def select(population):
            # Calculate the fitness of each individual
            fitness = np.array([func(individual) for individual in population])

            # Select the fittest individuals
            selected_individuals = np.argsort(fitness)[::-1][:self.population_size // 2]

            return selected_individuals

        # Define the mutation rate
        mutation_rate = 0.1

        # Define the crossover rate
        crossover_rate = 0.5

        # Define the selection rate
        selection_rate = 0.5

        # Initialize the population with the fittest individuals
        population = select(population)

        # Initialize the best individual and the best fitness
        best_individual = population[0]
        best_fitness = func(best_individual)

        # Initialize the number of generations
        generations = 0

        # Initialize the mutation counter
        mutation_counter = 0

        # Loop until the population converges or the maximum number of generations is reached
        while True:
            # Evaluate the function for each individual in the population
            func_values = func(population)

            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func_values)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            population = new_population

            # Update the best individual and the best fitness
            best_individual = population[0]
            best_fitness = func(best_individual)

            # Update the mutation counter
            mutation_counter += 1

            # Apply mutation to the population
            for individual in population:
                if random.random() < mutation_rate:
                    individual = mutate(individual)

            # Apply crossover to the population
            for i in range(0, population_size, 2):
                parent1 = population[i]
                parent2 = population[i + 1]

                # Apply crossover
                child = crossover(parent1, parent2)

                # Apply mutation to the child
                if random.random() < mutation_rate:
                    child = mutate(child)

                # Replace the old individual with the new individual
                population[i] = child
                population[i + 1] = child

            # Check for convergence
            if np.array_equal(population, best_individual) and best_fitness == func(best_individual):
                break

            # Increment the generation counter
            generations += 1

            # Update the best individual and the best fitness
            best_individual = population[0]
            best_fitness = func(best_individual)

        # Return the optimized parameters and the optimized function value
        return population, func(population)
