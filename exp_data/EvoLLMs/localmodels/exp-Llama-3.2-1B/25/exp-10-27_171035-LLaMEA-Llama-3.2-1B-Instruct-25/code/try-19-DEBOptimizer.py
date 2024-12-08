import numpy as np
from scipy.optimize import differential_evolution
from collections import deque
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
        self.population_history = deque(maxlen=self.budget)
        self.population_best = None
        self.population_best_fitness = float('-inf')

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

        # Update the population history
        self.population_history.append((func, results))

        # Update the population best and best fitness
        if len(self.population_history) > 1:
            self.population_best = self.population_history[-1][0]
            self.population_best_fitness = -self.population_history[-1][1][0].func(self.population_history[-1][1][0])
        else:
            self.population_best = None
            self.population_best_fitness = float('-inf')

        # Apply probability adjustment to the population
        if random.random() < 0.25:
            # Select a new individual by swapping two random individuals
            new_individuals = self.population_history[-1][1]
            i1, i2 = random.sample(range(len(new_individuals)), 2)
            new_individuals[i1], new_individuals[i2] = new_individuals[i2], new_individuals[i1]

            # Evaluate the new individual
            fitness_values = differential_evolution(lambda x: -func(x), new_individuals, bounds=(lower_bound, upper_bound), x0=new_individuals)

            # Replace the least fit individual with the new individual
            self.population_history[-1][1] = new_individuals
            self.population_history[-1][0] = func(self.population_history[-1][1][0])
            self.population_best = self.population_history[-1][0]
            self.population_best_fitness = -self.population_history[-1][1][0].func(self.population_history[-1][1][0])

        # Return the optimized function and its value
        return func(self.population_best), -func(self.population_best)

# One-line description with the main idea
# Evolutionary Black Box Optimization using Differential Evolution with Probability Adjustment