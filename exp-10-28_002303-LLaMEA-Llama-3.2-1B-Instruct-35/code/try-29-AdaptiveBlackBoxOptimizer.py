# Description: Adaptive Black Box Optimization Algorithm with Adaptive Sampling Strategy
# Code: 
# ```python
import random
import numpy as np

class AdaptiveBlackBoxOptimizer:
    """
    An optimization algorithm that uses adaptive search strategies to find the optimal solution.
    
    Parameters:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.sample_size = 1.0  # Initial sampling size
        self.best_solution = None
        self.best_cost = float('inf')

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
            # Initialize the current solution
            new_individual = np.random.uniform(-5.0, 5.0, self.dim)
            
            # Evaluate the function at the current solution
            cost = func(new_individual)
            
            # If the current solution is better than the best solution found so far, update the best solution
            if cost < best_cost:
                best_solution = new_individual
                best_cost = cost
            
            # Refine the sampling strategy
            if random.random() < 0.35:  # 35% chance to increase the sampling size
                self.sample_size *= 1.1  # Increase the sampling size by 10%
            
            # Update the best solution and its cost
            if cost < best_cost:
                best_solution = new_individual
                best_cost = cost
        
        # Return the optimal solution and its cost
        return best_solution, best_cost

# Description: Adaptive Black Box Optimization Algorithm with Adaptive Sampling Strategy
# Code: 
# ```python
# import numpy as np
# import random
# import time

def black_box_optimizer(budget, dim, sampling_strategy):
    """
    Optimize the black box function using the given budget and sampling strategy.
    
    Parameters:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    sampling_strategy (dict): A dictionary containing the sampling strategy, where keys are the sampling sizes and values are the corresponding sampling strategies.
    
    Returns:
    tuple: A tuple containing the optimal solution and its cost.
    """
    optimizer = AdaptiveBlackBoxOptimizer(budget, dim)
    func_evals = 0
    best_solution = None
    best_cost = float('inf')
    
    while True:
        # Optimize the function using the optimizer
        solution, cost = optimizer(func)
        
        # Increment the number of function evaluations
        func_evals += 1
        
        # If the number of function evaluations exceeds the budget, break the loop
        if func_evals > budget:
            break
        
        # Update the best solution and its cost based on the sampling strategy
        if sampling_strategy['sample_size'] > 1:  # Increase the sampling size by 10% when the sampling size is 1
            optimizer.sample_size *= 1.1  # Increase the sampling size by 10%
        
        # Update the best solution and its cost
        if cost < best_cost:
            best_solution = solution
            best_cost = cost
    
    return best_solution, best_cost

def main():
    budget = 1000
    dim = 10
    sampling_strategy = {'sample_size': 1.0}  # Initial sampling strategy with a sampling size of 1.0
    
    best_solution, best_cost = black_box_optimizer(budget, dim, sampling_strategy)
    print("Optimal solution:", best_solution)
    print("Optimal cost:", best_cost)

if __name__ == "__main__":
    main()