import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.best_individual = None
        self.best_value = float('-inf')

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

    def iterated_permutation_cooling(self, func, num_iterations):
        # Initialize the current best individual and value
        current_best_individual = None
        current_best_value = float('-inf')

        # Iterate over the number of iterations
        for _ in range(num_iterations):
            # Generate a random initial population of individuals
            population = np.random.uniform(-5.0, 5.0, (self.dim, self.budget))

            # Evaluate the population using the selected solution
            fitness_values = self.evaluate_fitness(population, func)

            # Select the individual with the highest fitness value
            current_individual = np.argmax(fitness_values)

            # Update the current best individual and value
            if fitness_values[current_individual] > current_best_value:
                current_best_individual = current_individual
                current_best_value = fitness_values[current_individual]

            # Perform the iterated permutation cooling step
            if random.random() < 0.45:
                # Select a random individual to swap with the current best individual
                swap_individual = np.random.choice(self.dim, 1, replace=False)

                # Swap the current best individual with the selected individual
                population[current_individual, swap_individual], population[swap_individual, current_individual] = population[swap_individual, current_individual], population[current_individual, swap_individual]

        # Return the best individual found
        return current_best_individual

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 