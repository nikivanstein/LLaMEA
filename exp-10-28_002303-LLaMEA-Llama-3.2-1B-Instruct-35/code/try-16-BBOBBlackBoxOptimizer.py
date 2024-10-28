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

def black_box_optimizer(budget, dim, strategy):
    """
    Optimizes the black box function using the given strategy.
    
    Parameters:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    strategy (str): The strategy to use for optimization.
    
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
    for _ in range(budget):
        # Generate a random solution within the search space
        solution = (lower_bound + random.uniform(-5.0, 5.0)) / 2
        
        # Evaluate the function at the current solution
        cost = func(solution)
        
        # If the current solution is better than the best solution found so far, update the best solution
        if cost < best_cost:
            best_solution = solution
            best_cost = cost
        
        # Apply the strategy to refine the solution
        if strategy == 'random':
            # Randomly swap the lower and upper bounds
            solution = (lower_bound + random.uniform(-5.0, 5.0)) / 2
        elif strategy == 'linear':
            # Scale the solution by a factor
            solution = (solution + 1) / 2
        elif strategy == 'logistic':
            # Scale the solution by a factor based on the cost
            solution = (1 / (1 + 10 ** (cost / 100))) * (lower_bound + upper_bound)
        else:
            # Raise the solution to a power based on the cost
            solution = (lower_bound + upper_bound) ** (cost / 100)
        
        # Evaluate the function at the current solution
        cost = func(solution)
        
        # If the current solution is better than the best solution found so far, update the best solution
        if cost < best_cost:
            best_solution = solution
            best_cost = cost
    
    # Return the optimal solution and its cost
    return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = black_box_optimizer(budget, dim, strategy='random')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost
# 
# def main():
#     budget = 1000
#     dim = 10
#     best_solution, best_cost = black_box_optimizer(budget, dim)
#     print("Optimal solution:", best_solution)
#     print("Optimal cost:", best_cost)
# 
# if __name__ == "__main__":
#     main()

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = black_box_optimizer(budget, dim, strategy='linear')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = black_box_optimizer(budget, dim, strategy='logistic')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = black_box_optimizer(budget, dim, strategy='mixed')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = black_box_optimizer(budget, dim, strategy='mixed')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = black_box_optimizer(budget, dim, strategy='mixed')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = black_box_optimizer(budget, dim, strategy='mixed')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = black_box_optimizer(budget, dim, strategy='adaptive')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = black_box_optimizer(budget, dim, strategy='adaptive')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = black_box_optimizer(budget, dim, strategy='mixed')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = black_box_optimizer(budget, dim, strategy='adaptive')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = black_box_optimizer(budget, dim, strategy='mixed')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = black_box_optimizer(budget, dim, strategy='adaptive')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = black_box_optimizer(budget, dim, strategy='mixed')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = black_box_optimizer(budget, dim, strategy='mixed')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = black_box_optimizer(budget, dim, strategy='adaptive')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = optimizer(func, strategy='mixed')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = optimizer(func, strategy='adaptive')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = optimizer(func, strategy='mixed')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = optimizer(func, strategy='adaptive')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = optimizer(func, strategy='mixed')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = optimizer(func, strategy='adaptive')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = optimizer(func, strategy='mixed')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = optimizer(func, strategy='adaptive')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = optimizer(func, strategy='mixed')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost

# Description: BBOB Black Box Optimizer with Evolutionary Strategy
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
        
#         # Apply the evolutionary strategy to refine the solution
#         solution, cost = optimizer(func, strategy='adaptive')
# 
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost