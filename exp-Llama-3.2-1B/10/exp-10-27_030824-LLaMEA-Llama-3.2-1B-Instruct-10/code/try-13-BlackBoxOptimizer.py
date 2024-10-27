import random
import numpy as np
from scipy.optimize import differential_evolution

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
        # Initialize the population with random points in the search space
        population = [np.random.uniform(self.search_space[0], self.search_space[1]) for _ in range(100)]
        
        # Evaluate the fitness of each individual in the population
        fitnesses = [func(individual) for individual in population]
        
        # Select the fittest individuals to reproduce
        fittest_individuals = np.argsort(fitnesses)[-budget:]
        
        # Create a new population by breeding the fittest individuals
        new_population = []
        for _ in range(budget):
            parent1, parent2 = random.sample(fittest_individuals, 2)
            child = (parent1 + parent2) / 2
            new_population.append(child)
        
        # Replace the old population with the new one
        population = new_population
        
        # Return the new population with the selected fittest individuals
        return population

# One-line description: Novel metaheuristic algorithm for black box optimization using a combination of random walk and linear interpolation.

# Code: