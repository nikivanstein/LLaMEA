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

            # Check if the population has reached the budget
            if len(self.population) > self.budget:
                break

        # Return the optimized function and its value
        return func(self.population[0]), -func(self.population[0])

    def adaptive_mutation(self, individual, mutation_rate):
        """
        Apply adaptive mutation to the individual.

        Args:
            individual (np.ndarray): The individual to mutate.
            mutation_rate (float): The probability of mutation.

        Returns:
            np.ndarray: The mutated individual.
        """
        # Get the bounds of the search space
        lower_bound = -5.0
        upper_bound = 5.0

        # Select a random index for mutation
        idx = random.randint(0, self.dim - 1)

        # Apply mutation
        mutated_individual = individual.copy()
        mutated_individual[idx] += np.random.uniform(-0.1, 0.1) * (upper_bound - lower_bound)

        # Ensure the mutated individual stays within the bounds
        mutated_individual[idx] = np.clip(mutated_individual[idx], lower_bound, upper_bound)

        return mutated_individual

    def differential_evolution_with_adaptive_mutation(self, func):
        """
        Optimize a black box function using differential evolution with adaptive mutation.

        Args:
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized function and its value.
        """
        # Initialize the population with random solutions
        self.population = [np.random.uniform(-5.0, 5.0, size=(100, self.dim)) for _ in range(100)]

        # Evaluate the objective function for each individual in the population
        results = []
        for _ in range(100):
            # Evaluate the objective function for each individual in the population
            fitness_values = differential_evolution(lambda x: -func(x), self.population, bounds=(lower_bound, upper_bound), x0=self.population)

            # Select the fittest individuals for the next generation
            fittest_individuals = [self.population[i] for i, _ in enumerate(results) if _ == fitness_values.x[0]]

            # Replace the least fit individuals with the fittest ones
            self.population = [self.population[i] for i in range(100) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Update the population with the fittest individuals
            self.population += [self.population[i] for i in range(100) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Check if the population has reached the budget
            if len(self.population) > self.budget:
                break

        # Apply adaptive mutation to the fittest individuals
        for individual in self.population:
            mutated_individual = self.adaptive_mutation(individual, 0.1)
            self.population[mutated_individual] = mutated_individual

        # Return the optimized function and its value
        return func(self.population[0]), -func(self.population[0])