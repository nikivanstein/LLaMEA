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

class GeneticProgrammingBBOBMetaheuristic(BBOBMetaheuristic):
    def __init__(self, budget, dim, mutation_rate):
        super().__init__(budget, dim)
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        # Define the search space
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
        # Initialize the solution
        sol = None
        
        # Try different initializations
        for _ in range(100):
            # Randomly initialize the solution
            sol = np.random.uniform(bounds, size=self.dim)
            
            # Evaluate the function at the solution
            func_sol = self.__call__(func, sol)
            
            # Check if the solution is better than the current best
            if func_sol < self.__call__(func, sol):
                # Update the solution
                sol = sol
        
        # Apply mutation to the solution
        for _ in range(self.budget):
            # Randomly select an individual
            individual = sol
            
            # Randomly select a mutation point
            mutation_point = random.randint(0, self.dim - 1)
            
            # Apply mutation
            individual[mutation_point] += random.uniform(-1, 1)
            
            # Update the solution
            sol = individual
        
        # Return the best solution found
        return sol

# One-line description with the main idea
# Genetic Programming BBOBMetaheuristic uses a genetic programming approach to optimize black box functions by iteratively refining the solution through mutation and selection.