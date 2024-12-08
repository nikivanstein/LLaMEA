import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

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

    def iterated_permutation_cooling(self, func, initial_population, cooling_rate, max_iter):
        # Initialize the population with random points in the search space
        population = initial_population
        for _ in range(max_iter):
            # Generate a new population by iterated permutation
            new_population = []
            for _ in range(len(population)):
                # Randomly select two points from the current population
                i, j = np.random.choice(len(population), 2, replace=False)
                # Swap the points to create a new individual
                new_population.append(np.concatenate((population[i], population[j])))
            # Evaluate the new population
            new_fitness_values = [func(individual) for individual in new_population]
            # Evaluate the old population
            old_fitness_values = [func(individual) for individual in population]
            # Calculate the new population's fitness
            new_fitness = np.mean(new_fitness_values)
            # Calculate the old population's fitness
            old_fitness = np.mean(old_fitness_values)
            # Update the population's fitness
            population = [individual for individual, fitness in zip(population, [new_fitness, old_fitness]) if fitness > old_fitness]
            # Apply cooling
            if np.random.rand() < cooling_rate:
                population = np.random.choice(population, size=len(population), replace=True)
        # Return the best individual in the final population
        return np.max(population)