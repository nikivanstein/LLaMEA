import numpy as np
from scipy.optimize import differential_evolution
import random

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
        
        # Refine the strategy based on the results
        if res.fun > -1e10:
            # If the objective value is not already optimal, refine the search space
            new_bounds = ((-5.0 + 1e-6, 5.0 - 1e-6), (-5.0 + 1e-6, 5.0 - 1e-6))
            new_x = x + random.uniform(-1e-6, 1e-6)
            new_res = differential_evolution(lambda new_x: -y, [(new_x, y)], x0=new_x, bounds=new_bounds)
            return new_res.x, -new_res.fun
        else:
            return x, -res.fun


# Description: A novel metaheuristic algorithm for solving black box optimization problems.
# The algorithm uses differential evolution to search for the optimal solution in the search space.
# It is designed to handle a wide range of tasks and can be tuned for different performance.