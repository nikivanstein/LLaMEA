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

    def search(self, func, bounds, mutation_prob=0.25):
        # Define the search space
        sol = None
        for _ in range(10):
            # Randomly initialize the solution
            sol = np.random.uniform(bounds, size=self.dim)
            
            # Evaluate the function at the solution
            func_sol = self.__call__(func, sol)
            
            # Check if the solution is better than the current best
            if func_sol < self.__call__(func, sol):
                # Update the solution
                sol = sol
        
        # Refine the solution using mutation
        for _ in range(self.dim):
            if np.random.rand() < mutation_prob:
                # Randomly select an individual from the search space
                new_individual = np.random.uniform(bounds, size=self.dim)
                
                # Evaluate the function at the new individual
                func_new_sol = self.__call__(func, new_individual)
                
                # Check if the new solution is better than the current best
                if func_new_sol < self.__call__(func, sol):
                    # Update the solution
                    sol = new_individual
        
        # Return the best solution found
        return sol

# One-line description with the main idea
# A novel heuristic algorithm that uses differential evolution to optimize black box functions, with a probability of mutation to refine the solution.