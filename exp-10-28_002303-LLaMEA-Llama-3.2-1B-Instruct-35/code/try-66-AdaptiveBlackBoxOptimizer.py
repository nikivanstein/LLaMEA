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
        self.individual_refining = False

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
        
        # Refine the individual using the adaptive strategy
        if self.individual_refining:
            # Calculate the Area over the Convergence Curve (AOCC) score
            aocc_score = self.calculate_aocc_score(best_solution, best_cost)
            
            # Refine the individual based on the AOCC score
            if aocc_score < 0.35:
                self.individual_refining = True
                # Refine the individual by swapping two random individuals
                new_individual = np.random.choice([best_solution, new_individual], size=self.dim, replace=False)
            else:
                self.individual_refining = False
        
        # Return the optimal solution and its cost
        return best_solution, best_cost

    def calculate_aocc_score(self, solution, cost):
        """
        Calculate the Area over the Convergence Curve (AOCC) score.
        
        Parameters:
        solution (numpy array): The current solution.
        cost (float): The current cost.
        
        Returns:
        float: The AOCC score.
        """
        # Calculate the AOCC score based on the number of function evaluations
        aocc_score = 1 - (1 / (2 * self.budget))
        return aocc_score

# Description: Adaptive Black Box Optimization Algorithm with Adaptive Individual Refining
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim):
#     optimizer = AdaptiveBlackBoxOptimizer(budget, dim)
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
        
#         # Refine the individual using the adaptive strategy
#         if random.random() < 0.35:
            # Refine the individual by swapping two random individuals
            new_individual = np.random.choice([best_solution, solution], size=dim, replace=True)
        # Return the optimal solution and its cost
        return best_solution, best_cost
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