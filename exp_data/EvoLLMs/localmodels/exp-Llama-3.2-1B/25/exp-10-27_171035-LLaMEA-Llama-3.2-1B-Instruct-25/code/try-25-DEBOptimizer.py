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

        # Update the population with adaptive mutation
        for i in range(population_size):
            # Select a random individual
            individual = self.population[i]

            # Evaluate the objective function for the individual
            fitness = -func(individual)

            # Select the fittest individuals for the next generation
            fittest_individuals = [self.population[j] for j, _ in enumerate(results) if _ == fitness]

            # Replace the least fit individuals with the fittest ones
            self.population = [self.population[j] for j in range(population_size) if j not in [i for i, _ in enumerate(results) if _ == fitness]]

            # Update the population with the fittest individuals
            self.population += [self.population[j] for j in range(population_size) if j not in [i for i, _ in enumerate(results) if _ == fitness]]

            # Update the individual with adaptive mutation
            mutated_individual = individual + random.uniform(-0.1, 0.1) * (individual - self.population[i])

            # Evaluate the objective function for the mutated individual
            mutated_fitness = -func(mutated_individual)

            # Check if the mutated individual is better than the current best
            if mutated_fitness < fitness:
                # Update the best individual
                self.population[i] = mutated_individual
                results[i] = mutated_fitness

        # Return the optimized function and its value
        return func(self.population[0]), -func(self.population[0])

# Example usage:
def test_func(x):
    return np.sum(x**2)

optimizer = DEBOptimizer(100, 10)
optimized_function, _ = optimizer(test_func)
print(f"Optimized function: {optimized_function}")
print(f"Optimized value: {optimized_function(1)}")