import numpy as np
from scipy.optimize import differential_evolution
import copy

class BBOBOptimizer:
    """
    A metaheuristic algorithm for solving black box optimization problems.
    
    The algorithm uses differential evolution to search for the optimal solution in the search space.
    It is designed to handle a wide range of tasks and can be tuned for different performance.
    """

    def __init__(self, budget, dim):
        """
        Initialize the optimizer with a budget and dimensionality.
        
        Args:
            budget (int): The number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimize a black box function using the given budget.
        
        Args:
            func (callable): The black box function to optimize.
        
        Returns:
            tuple: The optimal solution and the corresponding objective value.
        """
        # Create a grid of points in the search space
        x = np.linspace(-5.0, 5.0, self.dim)
        
        # Evaluate the black box function at each point
        y = func(x)
        
        # Perform the optimization using differential evolution
        res = differential_evolution(lambda x: -y, [(x, y)], x0=x, bounds=((None, None), (None, None)), n_iter=self.budget)
        
        # Create a copy of the initial solution
        new_individual = copy.deepcopy(res.x)
        
        # Perform adaptive line search
        for _ in range(self.budget):
            # Get the current and previous solutions
            curr_solution = res.x
            prev_solution = copy.deepcopy(curr_solution)
            
            # Evaluate the objective function at the current solution
            curr_value = -res.fun
            
            # Evaluate the objective function at the previous solution
            prev_value = -res.fun
            
            # Calculate the difference between the current and previous solutions
            diff = curr_value - prev_value
            
            # If the difference is less than a certain threshold, perform a mutation
            if diff < 0.01:
                # Generate a new solution by mutating the current solution
                new_solution = np.random.uniform(curr_solution)
                
                # Evaluate the objective function at the new solution
                new_value = -res.fun
                
                # If the new value is better than the previous value, update the solution
                if new_value > curr_value:
                    new_individual = copy.deepcopy(new_solution)
                    
            # If the difference is not less than the threshold, update the solution
            else:
                break
        
        # Return the optimal solution and the corresponding objective value
        return new_individual, -res.fun


# One-line description with the main idea
# Black Box Optimization using Differential Evolution with Adaptive Line Search
# This algorithm uses differential evolution to search for the optimal solution in the search space, with adaptive line search to refine the strategy.