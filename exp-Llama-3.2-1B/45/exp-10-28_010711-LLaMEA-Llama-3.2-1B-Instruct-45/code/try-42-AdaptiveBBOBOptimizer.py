import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveBBOBOptimizer:
    """
    An adaptive differential evolution metaheuristic algorithm for solving black box optimization problems.
    
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

        # Refine the solution based on the probabilistic refining strategy
        refines = np.random.choice([True, False], size=self.budget, p=[0.45, 0.55])
        refined_individual = np.zeros_like(x)
        for i in range(self.budget):
            if refines[i]:
                refined_individual[i] = x[i]
            else:
                refined_individual[i] = x[i] + np.random.normal(0, 0.1, self.dim)

        # Evaluate the refined solution
        refined_y = func(refined_individual)

        # Return the optimal solution and the corresponding objective value
        return refined_individual, -refined_y