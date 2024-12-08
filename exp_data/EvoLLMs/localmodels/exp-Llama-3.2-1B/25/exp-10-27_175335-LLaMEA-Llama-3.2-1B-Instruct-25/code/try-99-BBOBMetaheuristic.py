# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python
import numpy as np
import random
from copy import deepcopy

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

    def search(self, func, bounds):
        # Define the search space
        bounds = np.linspace(bounds[0], bounds[1], self.dim, endpoint=False)
        
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
        # Randomly mutate the individual
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        for i in range(self.dim):
            if random.random() < 0.25:
                # Change the value of the individual
                individual[i] = random.uniform(bounds[i], bounds[i] + 0.1)
        
        # Return the mutated individual
        return individual

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        child1 = np.random.uniform(bounds, size=self.dim)
        child2 = np.random.uniform(bounds, size=self.dim)
        
        # Select the crossover point
        crossover_point = random.randint(0, self.dim - 1)
        
        # Perform crossover
        child1[crossover_point] = parent1[crossover_point]
        child2[crossover_point] = parent2[crossover_point]
        
        # Return the child
        return child1, child2

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# to handle a wide range of tasks by adapting to changing fitness landscapes