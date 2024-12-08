import numpy as np
import random

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

    def mutate(self, sol):
        # Refine the solution by changing one line of the strategy
        new_sol = sol.copy()
        new_sol[random.randint(0, self.dim-1)] = random.uniform(-5.0, 5.0)
        return new_sol

    def __next__(self):
        # Select a parent using tournament selection
        parents = []
        for _ in range(20):
            parent = self.search(func)
            parents.append(parent)
        
        # Select the best parent and crossover
        best_parent = max(parents, key=self.func_evals)
        child = self.mutate(best_parent)
        
        # Replace the old solution with the new one
        self.func_evals += 1
        self.func_evals -= 1
        return child

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python
# import random
# import numpy as np
# import copy
# import time

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

#     def search(self, func):
#         # Define the search space
#         bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
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
        
#         # Return the best solution found
#         return sol

#     def mutate(self, sol):
#         # Refine the solution by changing one line of the strategy
#         new_sol = copy.deepcopy(sol)
#         new_sol[random.randint(0, self.dim-1)] = random.uniform(-5.0, 5.0)
#         return new_sol

#     def __next__(self):
#         # Select a parent using tournament selection
#         parents = []
#         for _ in range(20):
#             parent = self.search(func)
#             parents.append(parent)
        
#         # Select the best parent and crossover
#         best_parent = max(parents, key=self.func_evals)
#         child = self.mutate(best_parent)
        
#         # Replace the old solution with the new one
#         self.func_evals += 1
#         self.func_evals -= 1
#         return child

# def main():
#     # Set the parameters
#     budget = 100
#     dim = 10
#     func = lambda x: x**2
#     algorithm = BBOBMetaheuristic(budget, dim)
#     print(algorithm.__next__())

# main()