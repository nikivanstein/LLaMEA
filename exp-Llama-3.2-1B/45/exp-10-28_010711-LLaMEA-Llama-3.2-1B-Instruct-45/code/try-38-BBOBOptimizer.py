import numpy as np
from scipy.optimize import differential_evolution

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
        
        # Refine the individual's strategy based on the optimization result
        if res.success:
            # If the optimization was successful, refine the individual's strategy
            updated_individual = self.refine_strategy(res.x, res.fun)
        else:
            # If the optimization failed, keep the individual's current strategy
            updated_individual = res.x
        
        # Return the optimal solution and the corresponding objective value
        return updated_individual, -res.fun


    def refine_strategy(self, individual, objective_value):
        """
        Refine the individual's strategy based on the optimization result.
        
        Args:
            individual (list): The individual's current strategy.
            objective_value (float): The objective value obtained during optimization.
        
        Returns:
            list: The refined individual's strategy.
        """
        # Calculate the probability of changing the individual's strategy
        probability = 0.45
        
        # Generate a new individual with a modified strategy
        new_individual = individual.copy()
        for i in range(len(individual)):
            # Randomly decide whether to change the current strategy
            if np.random.rand() < probability:
                # Randomly choose a new strategy for the current dimension
                new_individual[i] = np.random.uniform(-5.0, 5.0)
        
        # Return the refined individual's strategy
        return new_individual