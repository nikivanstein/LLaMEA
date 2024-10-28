import numpy as np
import random

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

    def iterated_permutation_cooling(self, func, initial_population, cooling_rate, num_iterations):
        # Initialize the population with random points in the search space
        population = initial_population
        
        for _ in range(num_iterations):
            # Evaluate the fitness of each individual in the population
            fitnesses = [func(individual) for individual in population]
            
            # Select the fittest individuals to reproduce
            parents = random.sample(population, int(len(population) * cooling_rate))
            
            # Create a new population by combining the selected parents
            new_population = []
            for _ in range(len(parents)):
                parent1, parent2 = random.sample(parents, 2)
                child = (parent1 + parent2) / 2
                new_population.append(child)
            population = new_population
            
            # Update the population with the new individuals
            population.extend(new_population)
        
        # Select the fittest individuals to reproduce
        parents = random.sample(population, int(len(population) * cooling_rate))
        
        # Create a new population by combining the selected parents
        new_population = []
        for _ in range(len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child = (parent1 + parent2) / 2
            new_population.append(child)
        population = new_population
        
        # Evaluate the fitness of each individual in the new population
        fitnesses = [func(individual) for individual in population]
        
        # Select the fittest individuals to replace the old population
        old_indices = np.argsort(fitnesses)
        new_population = [population[i] for i in old_indices]
        
        # Update the old population with the new individuals
        population = new_population
        
        return population

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 