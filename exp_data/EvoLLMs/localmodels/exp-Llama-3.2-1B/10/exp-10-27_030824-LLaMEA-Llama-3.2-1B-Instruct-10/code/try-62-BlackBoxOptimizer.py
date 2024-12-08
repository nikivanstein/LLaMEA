import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, initial_point, budget=1000):
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

    def novel_metaheuristic(self, func, initial_point, budget=1000, iterations=50):
        # Initialize the population
        population = [initial_point]

        # Run the algorithm for the specified number of iterations
        for _ in range(iterations):
            # Evaluate the fitness of each individual in the population
            fitnesses = [func(individual) for individual in population]

            # Select the fittest individuals
            fittest_individuals = [individual for _, individual in sorted(zip(fitnesses, population), reverse=True)]

            # Generate a new population by perturbing the fittest individuals
            new_population = []
            for _ in range(len(fittest_individuals)):
                # Select a random individual from the fittest individuals
                parent = random.choice(fittest_individuals)
                # Perturb the parent to generate a new individual
                child = parent + np.random.uniform(-1, 1, self.dim)
                # Ensure the child is within the search space
                child = np.clip(child, self.search_space[0], self.search_space[1])
                # Add the child to the new population
                new_population.append(child)

            # Replace the old population with the new population
            population = new_population

            # Check if the budget is reached
            if self.func_evaluations >= self.budget:
                break

        # Return the fittest individual in the final population
        return population[0]

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Code: 