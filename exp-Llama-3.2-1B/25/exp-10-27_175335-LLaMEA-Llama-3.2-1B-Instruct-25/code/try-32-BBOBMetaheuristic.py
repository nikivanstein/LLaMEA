import numpy as np
import random
from scipy.optimize import differential_evolution

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        # Check if the function can be evaluated within the budget
        if self.func_evals >= self.budget:
            raise ValueError("Not enough evaluations left to optimize the function")

        # Evaluate the function within the budget
        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def search(self, func):
        # Define the search space
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
        # Initialize the solution
        sol = None
        
        # Try different initializations
        for _ in range(10):
            # Randomly initialize the solution
            sol = np.random.uniform(bounds, size=self.dim)
            
            # Evaluate the function at the solution
            func_sol = self.__call__(func, sol)
            
            # Check if the solution is better than the current best
            if func_sol < self.__call__(func, sol):
                # Update the solution
                sol = sol
        
        # Return the best solution found
        return sol

    def refine(self, func, initial, bounds, iterations):
        # Define the mutation and selection operators
        def mutation(individual, bounds):
            # Randomly mutate the individual
            mutated_individual = individual.copy()
            for i in range(len(individual)):
                if random.random() < 0.25:
                    mutated_individual[i] += random.uniform(-0.1, 0.1)
            return mutated_individual
        
        def selection(population, bounds):
            # Select the fittest individuals
            fittest_individuals = sorted(population, key=lambda x: x.fun, reverse=True)
            return fittest_individuals[:int(len(fittest_individuals) * 0.75)]

        # Run the evolutionary algorithm
        for _ in range(iterations):
            # Initialize the population
            population = [initial]
            
            # Evolve the population
            for _ in range(100):
                # Select the fittest individuals
                fittest_individuals = selection(population, bounds)
                
                # Mutate the fittest individuals
                mutated_individuals = [mutation(individual, bounds) for individual in fittest_individuals]
                
                # Replace the least fit individuals with the mutated ones
                population = [individual for individual in population if individual in mutated_individuals] + fittest_individuals
                
                # Evaluate the new population
                new_population = [self.search(func) for func in population]
                
                # Replace the old population with the new one
                population = new_population
        
        # Return the best solution found
        return population[0]

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming with Refinement
# This algorithm optimizes a black box function using a genetic programming approach with refinement, allowing for the refinement of the solution strategy based on the performance of the initial solution.

# Code