# Description: Evolutionary Black Box Optimization using Differential Evolution with Adaptive Mutation
# Code: 
# ```python
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
        self.mutation_rate = 0.1
        self.mutation_directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]

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

        # Return the optimized function and its value
        return func(self.population[0]), -func(self.population[0])

    def mutate(self, individual):
        """
        Mutate an individual in the population.

        Args:
            individual (numpy.ndarray): The individual to mutate.

        Returns:
            numpy.ndarray: The mutated individual.
        """
        # Select a random mutation direction
        direction = random.choice(self.mutation_directions)

        # Apply the mutation
        mutated_individual = individual.copy()
        mutated_individual[direction[0]] += random.uniform(-1, 1)
        mutated_individual[direction[1]] += random.uniform(-1, 1)

        return mutated_individual

    def update_population(self):
        """
        Update the population with new individuals.
        """
        # Evaluate the objective function for each individual in the population
        fitness_values = differential_evolution(lambda x: -func(x), self.population, bounds=(lower_bound, upper_bound), x0=self.population)

        # Select the fittest individuals for the next generation
        fittest_individuals = [self.population[i] for i, _ in enumerate(fitness_values.x) if _ == fitness_values.x[0]]

        # Replace the least fit individuals with the fittest ones
        self.population = [self.population[i] for i in range(len(self.population)) if i not in [j for j, _ in enumerate(fitness_values.x) if _ == fitness_values.x[0]]]

        # Update the population with the fittest individuals
        self.population += [self.population[i] for i in range(len(self.population)) if i not in [j for j, _ in enumerate(fitness_values.x) if _ == fitness_values.x[0]]]

        # Check if the population has reached the budget
        if len(self.population) > self.budget:
            break

        # Mutate the population
        for _ in range(int(len(self.population) * self.mutation_rate)):
            self.population.append(self.mutate(self.population[-1]))

        # Check if the population has reached the budget
        if len(self.population) > self.budget:
            break

        # Select the fittest individuals for the next generation
        fittest_individuals = [self.population[i] for i in range(len(self.population)) if i not in [j for j, _ in enumerate(fitness_values.x) if _ == fitness_values.x[0]]]

        # Replace the least fit individuals with the fittest ones
        self.population = [self.population[i] for i in range(len(self.population)) if i not in [j for j, _ in enumerate(fitness_values.x) if _ == fitness_values.x[0]]]

        # Update the population with the fittest individuals
        self.population += [self.population[i] for i in range(len(self.population)) if i not in [j for j, _ in enumerate(fitness_values.x) if _ == fitness_values.x[0]]]

        # Check if the population has reached the budget
        if len(self.population) > self.budget:
            break