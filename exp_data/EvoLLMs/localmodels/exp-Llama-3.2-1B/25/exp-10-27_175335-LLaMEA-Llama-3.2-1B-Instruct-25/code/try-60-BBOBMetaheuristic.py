import numpy as np
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

    def dynamic_line_refining(self, func, initial_solution, budget):
        # Define the search space
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
        # Initialize the solution
        sol = initial_solution
        
        # Try different initializations
        for _ in range(budget):
            # Randomly refine the solution
            new_solution = sol + np.random.uniform(-1, 1, size=self.dim)
            
            # Evaluate the function at the new solution
            func_sol = self.__call__(func, new_solution)
            
            # Check if the new solution is better than the current best
            if func_sol < self.__call__(func, sol):
                # Update the solution
                sol = new_solution
        
        # Return the best solution found
        return sol