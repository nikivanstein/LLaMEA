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

    def search(self, func, bounds, mutation_rate=0.01):
        # Define the search space
        sol = None
        for _ in range(self.dim):
            # Randomly initialize the solution
            sol = np.random.uniform(bounds, size=self.dim)
            
            # Evaluate the function at the solution
            func_sol = self.__call__(func, sol)
            
            # Check if the solution is better than the current best
            if func_sol < self.__call__(func, sol):
                # Update the solution
                sol = sol
                # Refine the solution strategy
                if random.random() < self.mutation_rate:
                    # Randomly swap two random dimensions
                    idx1, idx2 = random.sample(range(self.dim), 2)
                    sol[idx1], sol[idx2] = sol[idx2], sol[idx1]
        
        # Return the best solution found
        return sol

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Refines the solution strategy by changing the individual lines of the selected solution to refine its strategy
# with a probability of 0.25
# 
# Code:
# ```python
# import numpy as np
# import random
# import scipy.optimize as optimize
# import genetic

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

#     def search(self, func, bounds, mutation_rate=0.01):
#         # Define the search space
#         sol = None
#         for _ in range(self.dim):
#             # Randomly initialize the solution
#             sol = np.random.uniform(bounds, size=self.dim)
            
#             # Evaluate the function at the solution
#             func_sol = self.__call__(func, sol)
            
#             # Check if the solution is better than the current best
#             if func_sol < self.__call__(func, sol):
#                 # Update the solution
#                 sol = sol
                # Refine the solution strategy
                # with a probability of 0.25
                if random.random() < mutation_rate:
                    # Randomly swap two random dimensions
                    idx1, idx2 = random.sample(range(self.dim), 2)
                    sol[idx1], sol[idx2] = sol[idx2], sol[idx1]
        
#         # Return the best solution found
#         return sol

# mutation_rate = 0.01
# algorithm = BBOBMetaheuristic(100, 10)
# while True:
#     # Search for a new solution
#     new_individual = algorithm.search(lambda x: x[0]**2 + x[1]**2, [-10, 10], mutation_rate)
#     # Save the new solution
#     np.save(f"currentexp/aucs-{algorithm.__class__.__name__}-{i}.npy", new_individual)
#     # Update the best solution found
#     best_individual = max(algorithm.func(x) for x in range(100))
#     np.save(f"currentexp/aucs-{algorithm.__class__.__name__}-{i}.npy", best_individual)
#     # Update the algorithm
#     algorithm.func_evals = 0
#     algorithm.func_evals += 1
#     # Print the best solution found
#     print(f"Best solution found: {best_individual}")