import numpy as np
import random
import time

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

def adaptive_bbobOptimizer(budget, dim, num_iterations):
    """
    An adaptive version of the BBOB optimizer, which adjusts the number of function evaluations based on the performance of the solution.
    
    Parameters:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    num_iterations (int): The number of iterations to perform.
    
    Returns:
    tuple: A tuple containing the optimal solution and its cost.
    """
    # Initialize the best solution and its cost
    best_solution = None
    best_cost = float('inf')
    
    # Initialize the number of function evaluations
    func_evals = 0
    
    # Perform the given number of iterations
    for _ in range(num_iterations):
        # Optimize the function using the adaptive algorithm
        solution, cost = adaptiveOptimizer(budget, dim, func_evals)
        
        # Increment the number of function evaluations
        func_evals += 1
        
        # If the number of function evaluations exceeds the budget, break the loop
        if func_evals > budget:
            break
        
        # Update the best solution and its cost
        if cost < best_cost:
            best_solution = solution
            best_cost = cost
    
    # Return the optimal solution and its cost
    return best_solution, best_cost

def adaptiveOptimizer(budget, dim, func_evals):
    """
    An adaptive version of the BBOB optimizer, which adjusts the number of function evaluations based on the performance of the solution.
    
    Parameters:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    func_evals (int): The current number of function evaluations.
    
    Returns:
    tuple: A tuple containing the optimal solution and its cost.
    """
    # Initialize the search space
    lower_bound = -5.0
    upper_bound = 5.0
    
    # Initialize the best solution and its cost
    best_solution = None
    best_cost = float('inf')
    
    # Initialize the number of function evaluations
    updated_func_evals = 0
    
    # Initialize the number of iterations
    iterations = 0
    
    # Perform the given number of iterations
    while True:
        # Generate a random solution within the search space
        solution = (lower_bound + random.uniform(-5.0, 5.0)) / 2
        
        # Evaluate the function at the current solution
        cost = func(solution)
        
        # If the current solution is better than the best solution found so far, update the best solution
        if cost < best_cost:
            best_solution = solution
            best_cost = cost
        
        # Increment the number of function evaluations
        updated_func_evals += 1
        
        # If the number of function evaluations exceeds the budget, break the loop
        if updated_func_evals > budget:
            break
        
        # Increment the number of iterations
        iterations += 1
        
        # Update the best solution and its cost
        if cost < best_cost:
            best_solution = solution
            best_cost = cost
    
    # Return the optimal solution and its cost
    return best_solution, best_cost

# Description: Adaptive BBOB Optimizer
# Code: 