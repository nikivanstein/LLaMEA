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
        
        # Perform the optimization using differential evolution with adaptive line search
        res = differential_evolution(lambda x: -y, [(x, y)], x0=x, bounds=((None, None), (None, None)), n_iter=self.budget, maxiter=100, tol=1e-6, stepsize=self.stepsize(x, y))
        
        # Return the optimal solution and the corresponding objective value
        return res.x, -res.fun

    def stepsize(self, x, y):
        """
        Calculate the step size for the differential evolution algorithm.
        
        Args:
            x (float): The current point in the search space.
            y (float): The corresponding objective value.
        
        Returns:
            float: The step size.
        """
        # Calculate the step size based on the current point and objective value
        step_size = 0.1 * np.sqrt(y / (1 + y))
        return step_size

# Description: Evolutionary Optimization using Differential Evolution with Adaptive Line Search
# Code: 