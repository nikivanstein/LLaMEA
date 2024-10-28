# Description: Novel BBOB Black Box Optimizer with adaptive line search
# Code: 
# ```python
import random
import numpy as np

class BBOBBlackBoxOptimizer:
    """
    An optimization algorithm that uses black box function evaluations to find the optimal solution.
    
    Parameters:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        """
        Optimize the black box function using the given budget for function evaluations.
        
        Parameters:
        func (function): The black box function to optimize.
        
        Returns:
        tuple: A tuple containing the optimal solution and its cost.
        """
        # Initialize the search space
        lower_bound = -5.0
        upper_bound = 5.0
        
        # Initialize the best solution and its cost
        best_solution = None
        best_cost = float('inf')
        
        # Perform the given number of function evaluations
        for _ in range(self.budget):
            # Generate a random solution within the search space
            solution = (lower_bound + random.uniform(-5.0, 5.0)) / 2
            
            # Evaluate the function at the current solution
            cost = func(solution)
            
            # If the current solution is better than the best solution found so far, update the best solution
            if cost < best_cost:
                best_solution = solution
                best_cost = cost
        
        # Return the optimal solution and its cost
        return best_solution, best_cost

# Description: Novel BBOB Black Box Optimizer with adaptive line search
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim):
#     optimizer = BBOBBlackBoxOptimizer(budget, dim)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
# 
#         # Increment the number of function evaluations
#         func_evals += 1
# 
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost
# 
# def adaptive_line_search(optimizer, func, initial_solution, alpha=0.1, max_iter=1000):
#     """
    #     Apply an adaptive line search to the optimizer.
    
    #     Parameters:
    #     optimizer (BBOBBlackBoxOptimizer): The optimizer to apply the line search to.
    #     func (function): The black box function to optimize.
    #     initial_solution (list): The initial solution to use for the line search.
    #     alpha (float, optional): The step size for the line search. Defaults to 0.1.
    #     max_iter (int, optional): The maximum number of iterations for the line search. Defaults to 1000.
    #     """
#     for _ in range(max_iter):
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
        
#         # Update the best solution and its cost
#         if cost < initial_solution[-1]:
            # initial_solution = solution
            # best_solution = solution
            # best_cost = cost
        
#         # Apply the adaptive line search
#         for i in range(len(initial_solution) - 1):
#             alpha_ = alpha * (1 + (1 - alpha) * (solution[i] - initial_solution[i]) / (initial_solution[i + 1] - solution[i]))
#             alpha_ = max(alpha_, 1e-8)
#             initial_solution[i + 1] = solution[i] + alpha_ * (solution[i + 1] - solution[i])
        
#         # Update the best solution and its cost
#         if cost < initial_solution[-1]:
            # best_solution = initial_solution
            # best_cost = cost
        
#     return best_solution, best_cost

# Description: Novel BBOB Black Box Optimizer with adaptive line search
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim):
#     optimizer = BBOBBlackBoxOptimizer(budget, dim)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
# 
#         # Increment the number of function evaluations
#         func_evals += 1
# 
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: Novel BBOB Black Box Optimizer with adaptive line search
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim):
#     optimizer = BBOBBlackBoxOptimizer(budget, dim)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
# 
#         # Increment the number of function evaluations
#         func_evals += 1
# 
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: Novel BBOB Black Box Optimizer with adaptive line search
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim):
#     optimizer = BBOBBlackBoxOptimizer(budget, dim)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
# 
#         # Increment the number of function evaluations
#         func_evals += 1
# 
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Apply adaptive line search
#         if func_evals > 100:
#             alpha = 0.1
#         else:
#             alpha = 0.01
#         best_solution, best_cost = adaptive_line_search(optimizer, func, [best_solution] * 10, alpha=alpha, max_iter=1000)
        
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: Novel BBOB Black Box Optimizer with adaptive line search
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim):
#     optimizer = BBOBBlackBoxOptimizer(budget, dim)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
# 
#         # Increment the number of function evaluations
#         func_evals += 1
# 
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Apply adaptive line search
#         alpha = 0.1
#         best_solution, best_cost = adaptive_line_search(optimizer, func, [best_solution] * 10, alpha=alpha, max_iter=1000)
        
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: Novel BBOB Black Box Optimizer with adaptive line search
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim):
#     optimizer = BBOBBlackBoxOptimizer(budget, dim)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
# 
#         # Increment the number of function evaluations
#         func_evals += 1
# 
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: Novel BBOB Black Box Optimizer with adaptive line search
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim):
#     optimizer = BBOBBlackBoxOptimizer(budget, dim)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
# 
#         # Increment the number of function evaluations
#         func_evals += 1
# 
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost