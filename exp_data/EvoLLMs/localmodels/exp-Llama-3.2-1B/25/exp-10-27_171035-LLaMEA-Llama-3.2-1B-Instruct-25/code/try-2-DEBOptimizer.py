# Description: Evolutionary Black Box Optimization using Differential Evolution with Adaptive Mutation Strategy
# Code: 
# import numpy as np
# import random
# import math

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

            # Check if the population has reached the budget
            if len(self.population) > self.budget:
                break

        # Apply adaptive mutation strategy
        self.population = self.apply_adaptive Mutation(self.population)

        # Return the optimized function and its value
        return func(self.population[0]), -func(self.population[0])

    def apply_adaptiveMutation(self, population):
        """
        Apply adaptive mutation strategy to the population.

        Args:
            population (list): The population of individuals.

        Returns:
            list: The mutated population.
        """
        # Initialize the mutation rate and the mutation probability
        mutation_rate = 0.1
        mutation_probability = 0.5

        # Initialize the mutated population
        mutated_population = population.copy()

        # Iterate over the population
        for i in range(len(population)):
            # Get the current individual
            individual = population[i]

            # Generate a mutation vector
            mutation_vector = np.random.uniform(-1, 1, size=self.dim)

            # Apply mutation to the individual
            mutated_individual = individual.copy()
            for j in range(self.dim):
                mutated_individual[j] += mutation_vector[j] * math.sqrt(mutation_rate)

            # Check if the mutation is valid
            if random.random() < mutation_probability:
                # Update the individual with a new mutation
                mutated_individual[j] += random.uniform(-5, 5)
                mutated_individual[j] = max(-5, min(5, mutated_individual[j]))

            # Replace the individual with the mutated individual
            mutated_population[i] = mutated_individual

        # Return the mutated population
        return mutated_population