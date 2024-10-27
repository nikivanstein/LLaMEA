# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
import numpy as np
import random
import copy

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

    def mutate(self, individual):
        # Refine the individual's strategy by changing 25% of its lines
        mutated_individual = copy.deepcopy(individual)
        for i in range(len(individual)):
            if random.random() < 0.25:
                mutated_individual[i] = random.uniform(-5.0, 5.0)
        return mutated_individual

# One-line description with main idea
# "Evolutionary Algorithm for Black Box Optimization using Genetic Programming"