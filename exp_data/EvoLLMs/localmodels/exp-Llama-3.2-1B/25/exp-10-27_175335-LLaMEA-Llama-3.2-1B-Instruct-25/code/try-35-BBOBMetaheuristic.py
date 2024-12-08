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

    def search(self, func, mutation_rate, bounds, mutation_threshold):
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
            
            # Apply mutation to the solution
            if np.random.rand() < mutation_rate:
                # Randomly select an element from the current solution
                idx = np.random.randint(0, self.dim)
                
                # Apply mutation to the element
                sol[idx] = np.random.uniform(bounds[idx])
                
                # Check if the mutation is within the threshold
                if np.abs(sol[idx] - bounds[idx]) < mutation_threshold:
                    # Update the solution
                    sol[idx] = bounds[idx]
        
        # Return the best solution found
        return sol

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python
# import numpy as np
# import scipy.optimize as optimize

# class BBOBMetaheuristic:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.func_evals = 0

#     def __call__(self, func):
#         # Check if the function can be evaluated within the budget
#         if self.func_evals >= self.budget:
#             raise ValueError("Not enough evaluations left to optimize the function")

#         # Evaluate the function within the budget
#         func_evals = self.func_evals
#         self.func_evals += 1
#         return func

#     def search(self, func, mutation_rate, bounds, mutation_threshold):
#         # Define the search space
#         bounds = np.linspace(bounds[0], bounds[1], self.dim, endpoint=False)
        
#         # Initialize the solution
#         sol = None
        
#         # Try different initializations
#         for _ in range(10):
#             # Randomly initialize the solution
#             sol = np.random.uniform(bounds, size=self.dim)
            
#             # Evaluate the function at the solution
#             func_sol = self.__call__(func, sol)
            
#             # Check if the solution is better than the current best
#             if func_sol < self.__call__(func, sol):
#                 # Update the solution
#                 sol = sol
            
#             # Apply mutation to the solution
#             if np.random.rand() < mutation_rate:
#                 # Randomly select an element from the current solution
#                 idx = np.random.randint(0, self.dim)
                
#                 # Apply mutation to the element
#                 sol[idx] = np.random.uniform(bounds[idx])
                
#                 # Check if the mutation is within the threshold
#                 if np.abs(sol[idx] - bounds[idx]) < mutation_threshold:
#                     # Update the solution
#                     sol[idx] = bounds[idx]
        
#         # Return the best solution found
#         return sol

def mutation_exp(budget, dim, mutation_rate, bounds, mutation_threshold):
    algorithm = BBOBMetaheuristic(budget, dim)
    solution = algorithm.search(lambda func: func(np.random.uniform(-5.0, 5.0, dim)), mutation_rate, bounds, mutation_threshold)
    return solution

# Test the function
budget = 1000
dim = 10
mutation_rate = 0.01
bounds = np.linspace(-5.0, 5.0, dim)
mutation_threshold = 0.1
solution = mutation_exp(budget, dim, mutation_rate, bounds, mutation_threshold)
print(solution)