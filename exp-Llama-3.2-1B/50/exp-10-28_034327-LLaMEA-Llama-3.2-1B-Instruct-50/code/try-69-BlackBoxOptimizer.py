import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.current_individual = None

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            self.current_individual = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(self.current_individual)
            # Check if the point is within the bounds
            if -5.0 <= self.current_individual[0] <= 5.0 and -5.0 <= self.current_individual[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

    def iterated_permutation(self, func, num_iterations):
        for _ in range(num_iterations):
            # Initialize the population with the current individual
            population = [self.current_individual]
            # Generate new individuals by iterating over the population
            for _ in range(self.budget):
                # Generate a new individual by perturbing the current individual
                new_individual = self.current_individual + np.random.uniform(-5.0, 5.0, self.dim)
                # Check if the new individual is within the bounds
                if -5.0 <= new_individual[0] <= 5.0 and -5.0 <= new_individual[1] <= 5.0:
                    # If the new individual is within bounds, add it to the population
                    population.append(new_individual)
            # Select the fittest individuals for the next iteration
            population = sorted(population, key=lambda individual: func(individual), reverse=True)[:self.budget]
            # Update the current individual
            self.current_individual = population[0]
        # Return the best individual found
        return self.current_individual

    def cooling_algorithm(self, func, num_iterations):
        for _ in range(num_iterations):
            # Initialize the population with the current individual
            population = [self.current_individual]
            # Generate new individuals by iterating over the population
            for _ in range(self.budget):
                # Generate a new individual by perturbing the current individual
                new_individual = self.current_individual + np.random.uniform(-5.0, 5.0, self.dim)
                # Check if the new individual is within the bounds
                if -5.0 <= new_individual[0] <= 5.0 and -5.0 <= new_individual[1] <= 5.0:
                    # If the new individual is within bounds, add it to the population
                    population.append(new_individual)
            # Select the fittest individuals for the next iteration
            population = sorted(population, key=lambda individual: func(individual), reverse=True)
            # Update the current individual
            self.current_individual = population[0]
        # Return the best individual found
        return self.current_individual