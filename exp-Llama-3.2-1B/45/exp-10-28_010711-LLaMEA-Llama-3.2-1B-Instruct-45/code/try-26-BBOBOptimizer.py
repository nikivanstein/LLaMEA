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
        
        # Refine the strategy by changing the individual lines of the selected solution
        for i in range(self.budget):
            if random.random() < 0.45:
                # Change the lower bound of the search space
                x = np.linspace(-5.0, 5.0, self.dim)
                y = func(x)
                res = differential_evolution(lambda x: -y, [(x, y)], x0=x, bounds=((None, None), (None, None)), n_iter=self.budget)
                
                # Change the upper bound of the search space
                x = np.linspace(5.0, 5.0, self.dim)
                y = func(x)
                res = differential_evolution(lambda x: -y, [(x, y)], x0=x, bounds=((None, None), (None, None)), n_iter=self.budget)
                
                # Change the individual lines of the selected solution
                x = np.linspace(-5.0, 5.0, self.dim)
                y = func(x)
                res = differential_evolution(lambda x: -y, [(x, y)], x0=x, bounds=((None, None), (None, None)), n_iter=self.budget)
                
                # Change the lower bound of the search space
                x = np.linspace(5.0, 5.0, self.dim)
                y = func(x)
                res = differential_evolution(lambda x: -y, [(x, y)], x0=x, bounds=((None, None), (None, None)), n_iter=self.budget)
                
                # Change the upper bound of the search space
                x = np.linspace(-5.0, 5.0, self.dim)
                y = func(x)
                res = differential_evolution(lambda x: -y, [(x, y)], x0=x, bounds=((None, None), (None, None)), n_iter=self.budget)
                
        # Return the optimal solution and the corresponding objective value
        return res.x, -res.fun


# Example usage:
optimizer = BBOBOptimizer(10, 5)
func = lambda x: np.sin(x)
optimal_solution, objective_value = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Objective value:", objective_value)