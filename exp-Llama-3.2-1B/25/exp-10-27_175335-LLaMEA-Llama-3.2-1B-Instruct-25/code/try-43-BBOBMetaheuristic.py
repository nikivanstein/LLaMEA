import numpy as np
import random

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.line_lengths = [5.0]  # Initialize line lengths with a fixed value

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
            
            # Refine the solution based on the fitness value
            if func_sol < self.__call__(func, sol):
                # Increase the line length of the solution
                self.line_lengths.append(max(self.line_lengths[-1] + 0.1, 10.0))
                
                # Update the solution
                sol = np.random.uniform(bounds, size=self.dim)
                
                # Evaluate the function at the new solution
                func_sol = self.__call__(func, sol)
                
                # Check if the new solution is better than the current best
                if func_sol < self.__call__(func, sol):
                    # Update the solution
                    sol = sol
        
        # Return the best solution found
        return sol

# Example usage
bboo = BBOBMetaheuristic(100, 10)
func = lambda x: x**2  # Black box function to optimize
best_solution = bboo.search(func)
print("Best solution:", best_solution)