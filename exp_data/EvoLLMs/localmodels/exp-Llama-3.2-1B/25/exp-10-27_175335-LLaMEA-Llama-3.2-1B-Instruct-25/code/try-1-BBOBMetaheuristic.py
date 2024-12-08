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
            sol = copy.deepcopy(bounds)
            
            # Refine the solution using the probability 0.25 to change the individual lines of the selected solution
            for _ in range(5):
                # Randomly choose a line to change
                line_to_change = random.randint(0, self.dim - 1)
                
                # Change the line to change
                sol[line_to_change] += random.uniform(-0.1, 0.1)
                
                # Update the solution
                sol = copy.deepcopy(bounds)
                for i in range(self.dim):
                    if i!= line_to_change:
                        sol[i] += random.uniform(-0.1, 0.1)
                
                # Evaluate the function at the new solution
                func_sol = self.__call__(func, sol)
                
                # Check if the solution is better than the current best
                if func_sol < self.__call__(func, sol):
                    # Update the solution
                    sol = sol
        
        # Return the best solution found
        return sol

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 