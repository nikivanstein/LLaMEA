import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.best_individual = None
        self.best_fitness = -np.inf

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

    def iterated_permutation_cooling(self, func, initial_individual, initial_fitness):
        # Initialize the current best individual and fitness
        current_individual = initial_individual
        current_fitness = initial_fitness

        # Initialize the cooling schedule
        cooling_schedule = [0.1, 0.5, 0.9]

        # Initialize the population
        population = [current_individual]

        # Iterate until the budget is exceeded
        for _ in range(self.budget):
            # Evaluate the fitness of each individual in the population
            fitnesses = [func(individual) for individual in population]

            # Select the fittest individual
            fittest_individual = population[np.argmax(fitnesses)]

            # Update the current individual and fitness
            current_individual = fittest_individual
            current_fitness = fitnesses[np.argmax(fitnesses)]

            # Select a new individual using the iterated permutation
            if np.random.rand() < 0.1:  # 10% chance of using iterated permutation
                new_individual = fittest_individual
            else:
                # Use the cooling schedule to select the next individual
                index = np.random.randint(0, len(population))
                new_individual = population[index]

            # Add the new individual to the population
            population.append(new_individual)

            # Update the best individual and fitness
            if current_fitness < self.best_fitness:
                self.best_individual = current_individual
                self.best_fitness = current_fitness

        # Return the best individual found
        return self.best_individual

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 