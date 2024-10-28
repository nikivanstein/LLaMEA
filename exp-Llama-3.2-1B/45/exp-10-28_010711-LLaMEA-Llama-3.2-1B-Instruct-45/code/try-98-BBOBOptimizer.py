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
        
        # Refine the search space using the selected solution
        new_individual = self.evaluate_fitness(res.x)
        updated_individual = self.evaluate_fitness(new_individual)
        new_bounds = self.refine_bounds(res.x, updated_individual)
        
        # Return the optimal solution and the corresponding objective value
        return new_individual, -res.fun, new_bounds

    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual in the search space.
        
        Args:
            individual (list): The individual to evaluate.
        
        Returns:
            float: The fitness of the individual.
        """
        # Create a grid of points in the search space
        x = np.linspace(-5.0, 5.0, self.dim)
        
        # Evaluate the black box function at each point
        y = func(x)
        
        # Calculate the fitness as the negative of the objective value
        fitness = -y
        
        # Return the fitness of the individual
        return fitness

    def refine_bounds(self, x, new_individual):
        """
        Refine the search space using the selected solution.
        
        Args:
            x (list): The current search space.
            new_individual (list): The selected solution.
        
        Returns:
            tuple: The refined search space.
        """
        # Calculate the bounds of the new individual
        lower_bound = np.min(x)
        upper_bound = np.max(x)
        
        # Refine the bounds using the selected solution
        refined_bounds = (lower_bound, upper_bound)
        
        # Return the refined search space
        return refined_bounds