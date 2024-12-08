import numpy as np
import random

class GeneticBBOOPOptimizer:
    def __init__(self, budget, dim, alpha=0.25, beta=0.25):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.alpha = alpha
        self.beta = beta
        self.best_solution = None
        self.best_fitness = np.inf

    def __call__(self, func):
        if self.func_evals >= self.budget:
            raise ValueError("Not enough evaluations left to optimize the function")

        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def search(self, func):
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        sol = None
        for _ in range(10):
            sol = np.random.uniform(bounds, size=self.dim)
            func_sol = self.__call__(func, sol)
            if func_sol < self.__call__(func, sol):
                sol = sol
        
        # Update the best solution and fitness
        self.best_solution = sol
        self.best_fitness = np.linalg.norm(self.best_solution - sol)
        
        # Apply adaptive line search
        if self.best_fitness < self.best_solution:
            self.alpha = self.alpha * self.beta
            self.best_solution = sol

        # Return the best solution found
        return sol

# Description: Genetic Black Box Optimization with Adaptive Line Search
# Code: 
# ```python
# import numpy as np
# import random

# class GeneticBBOOPOptimizer:
#     def __init__(self, budget, dim, alpha=0.25, beta=0.25):
#         self.budget = budget
#         self.dim = dim
#         self.func_evals = 0
#         self.alpha = alpha
#         self.beta = beta
#         self.best_solution = None
#         self.best_fitness = np.inf

#     def __call__(self, func):
#         if self.func_evals >= self.budget:
#             raise ValueError("Not enough evaluations left to optimize the function")

#         func_evals = self.func_evals
#         self.func_evals += 1
#         return func

#     def search(self, func):
#         bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
#         sol = None
#         for _ in range(10):
#             sol = np.random.uniform(bounds, size=self.dim)
#             func_sol = self.__call__(func, sol)
#             if func_sol < self.__call__(func, sol):
#                 sol = sol
        
#         # Update the best solution and fitness
#         self.best_solution = sol
#         self.best_fitness = np.linalg.norm(self.best_solution - sol)
        
#         # Apply adaptive line search
#         if self.best_fitness < self.best_solution:
#             self.alpha = self.alpha * self.beta
#             self.best_solution = sol

#         # Return the best solution found
#         return sol

# optimizer = GeneticBBOOPOptimizer(budget=1000, dim=10)
# optimizer.search(lambda x: np.sin(x))