import numpy as np
import random

class BBOBRefiningOptimizer:
    """
    An optimization algorithm that uses black box function evaluations to find the optimal solution with a refining strategy.
    
    Parameters:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.refining_strategy = 0.35  # Probability of refining the individual lines

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
        
        # Refine the solution if necessary
        if self.refining_strategy > 0.0:
            # Select a random individual line from the search space
            line = random.choice([i for i in range(self.dim) if i not in [0, self.dim - 1]])
            
            # Optimize the selected individual line using the optimizer
            solution = (lower_bound + random.uniform(-5.0, 5.0)) / 2
            
            # Evaluate the function at the optimized individual line
            cost = func(solution)
            
            # If the function value is improved, update the best solution
            if cost < best_cost:
                best_solution = solution
                best_cost = cost
        
        # Return the optimal solution and its cost
        return best_solution, best_cost

# Description: Black Box Optimizer with Refining Strategy
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim):
#     optimizer = BBOBRefiningOptimizer(budget, dim)
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
# def main():
#     budget = 1000
#     dim = 10
#     best_solution, best_cost = black_box_optimizer(budget, dim)
#     print("Optimal solution:", best_solution)
#     print("Optimal cost:", best_cost)
# 
# if __name__ == "__main__":
#     main()