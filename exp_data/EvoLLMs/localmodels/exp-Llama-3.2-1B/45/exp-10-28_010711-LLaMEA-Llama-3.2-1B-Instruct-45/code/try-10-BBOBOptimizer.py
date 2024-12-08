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
        
        # If the optimization was successful, refine the solution with adaptive mutation strategy
        if res.success:
            # Get the current solution and its fitness
            current_solution = res.x
            current_fitness = -res.fun
            
            # Refine the solution by mutating a small portion of it
            mutated_solution = current_solution + random.uniform(-0.1, 0.1)
            
            # Evaluate the new solution
            new_fitness = -np.array([func(i) for i in mutated_solution]).mean()
            
            # Update the solution and its fitness
            new_solution = mutated_solution
            new_fitness = new_fitness
            
            # Update the best solution found so far
            if new_fitness > current_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
            else:
                best_solution = current_solution
                best_fitness = current_fitness
            
            # Return the refined solution and its updated fitness
            return best_solution, best_fitness
        
        # If the optimization failed, return the current solution and its fitness
        else:
            return res.x, -res.fun


# Example usage:
# optimizer = BBOBOptimizer(100, 10)
# func = lambda x: x[0]**2 + x[1]**2
# best_solution, best_fitness = optimizer(func)
# print("Best solution:", best_solution)
# print("Best fitness:", best_fitness)