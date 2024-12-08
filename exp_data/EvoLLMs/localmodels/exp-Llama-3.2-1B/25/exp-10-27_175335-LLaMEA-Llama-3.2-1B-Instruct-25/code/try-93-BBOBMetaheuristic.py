import numpy as np
import random
import time

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
            else:
                # Refine the solution by changing individual lines
                for i in range(self.dim):
                    # Randomly select an individual line to change
                    line = random.choice([0, 1])
                    
                    # Randomly change the individual line
                    if line == 0:
                        sol[i] = random.uniform(-5.0, 5.0)
                    else:
                        sol[i] = random.uniform(5.0, 5.0)
                
                # Evaluate the function at the new solution
                func_sol = self.__call__(func, sol)
                
                # Check if the new solution is better than the current best
                if func_sol < self.__call__(func, sol):
                    # Update the solution
                    sol = sol
        
        # Return the best solution found
        return sol

# One-Liner Description:
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming: A novel algorithm that refines its strategy by changing individual lines of the selected solution to optimize the function within the budget.

# Code: