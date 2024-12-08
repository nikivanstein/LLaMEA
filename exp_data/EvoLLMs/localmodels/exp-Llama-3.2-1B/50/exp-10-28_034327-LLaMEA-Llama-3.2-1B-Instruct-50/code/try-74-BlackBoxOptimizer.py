import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.best_individual = None
        self.best_fitness = float('-inf')

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

    def iterated_permutation_cooling(self, func):
        # Initialize the population with random points
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]
        
        # Initialize the best individual and its fitness
        self.best_individual = population[0]
        self.best_fitness = np.max(func(population[0]))
        
        # Iterate until the budget is exceeded
        for _ in range(self.budget):
            # Evaluate the fitness of each individual
            fitness = [func(individual) for individual in population]
            
            # Select the fittest individuals
            fittest_individuals = population[np.argsort(fitness)]
            
            # Create a new population by iterating over the fittest individuals
            new_population = []
            for _ in range(len(fittest_individuals)):
                # Generate a new point by iterated permutation and cooling
                new_point = fittest_individuals[_]
                for _ in range(self.dim):
                    new_point = new_point + random.uniform(-0.1, 0.1)
                new_point = new_point / np.linalg.norm(new_point)
                
                # Add the new point to the new population
                new_population.append(new_point)
            
            # Replace the old population with the new population
            population = new_population
            
            # Update the best individual and its fitness
            self.best_individual = population[0]
            self.best_fitness = np.max(func(self.best_individual))
        
        # Return the best individual found
        return self.best_individual

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 