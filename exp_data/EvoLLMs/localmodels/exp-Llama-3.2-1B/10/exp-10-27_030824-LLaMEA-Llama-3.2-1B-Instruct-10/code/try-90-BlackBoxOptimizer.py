import random
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def novel_metaheuristic(self, func, budget, dim):
        # Define the mutation function
        def mutate(individual):
            # Randomly select a point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Randomly select a direction to mutate
            direction = np.random.uniform(-1, 1, self.dim)
            # Update the individual using linear interpolation
            updated_individual = individual + evaluation * direction
            # Clip the updated individual to the search space
            updated_individual = np.clip(updated_individual, self.search_space[0], self.search_space[1])
            return updated_individual

        # Initialize the population
        population = [random.uniform(self.search_space[0], self.search_space[1]) for _ in range(100)]

        # Run the evolution
        for _ in range(1000):
            # Evaluate the fitness of each individual
            fitness = [func(individual) for individual in population]
            # Select the fittest individuals
            fittest_individuals = np.argsort(fitness)[-self.budget:]

            # Mutate the fittest individuals
            mutated_individuals = [mutate(individual) for individual in fittest_individuals]

            # Replace the least fit individuals with the mutated ones
            population[fittest_individuals] = mutated_individuals

            # Evaluate the fitness of each individual again
            fitness = [func(individual) for individual in population]
            # Select the fittest individuals again
            fittest_individuals = np.argsort(fitness)[-self.budget:]

            # Replace the least fit individuals with the fittest ones
            population[fittest_individuals] = population[fittest_individuals]

        # Return the fittest individual
        return population[0]

# One-line description: Novel metaheuristic algorithm for black box optimization using a combination of random walk and linear interpolation.
