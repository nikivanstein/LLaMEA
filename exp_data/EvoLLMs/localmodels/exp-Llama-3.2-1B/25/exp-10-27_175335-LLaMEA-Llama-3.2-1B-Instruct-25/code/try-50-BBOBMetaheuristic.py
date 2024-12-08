import numpy as np
import random
from typing import Dict

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population = []
        self.population_history = {}

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

    def mutate(self, individual):
        # Randomly change a single element in the individual
        index = random.randint(0, self.dim - 1)
        new_individual = individual.copy()
        new_individual[index] = random.uniform(-5.0, 5.0)
        return new_individual

    def adapt(self, new_individual):
        # Update the population with a new individual
        self.population.append(new_individual)
        self.population_history[new_individual] = self.func_evals

    def __next__(self):
        # Select the best individual from the population
        best_individual = min(self.population, key=self.func_evals)
        
        # Generate a new individual by adapting the best individual
        new_individual = self.adapt(best_individual)
        
        # Update the best individual
        best_individual = new_individual
        
        # Return the new individual
        return best_individual

# One-line description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with Adaptation
# Code: 