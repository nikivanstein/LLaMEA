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
        
        # Refine the strategy using a metaheuristic algorithm
        if res.success:
            # Use a simple mutation strategy to refine the individual lines of the selected solution
            updated_individual = self.evaluate_fitness(res.x)
            updated_individual = self._refine_individual(updated_individual)
            return updated_individual, -res.fun
        else:
            # If the optimization fails, return the original individual and the objective value
            return res.x, -res.fun

    def _refine_individual(self, individual):
        """
        Refine the individual lines of the selected solution using a metaheuristic algorithm.
        
        Args:
            individual (list): The selected solution.
        
        Returns:
            list: The refined individual.
        """
        # Define a simple mutation strategy
        mutation_rate = 0.1
        mutation_individual = individual.copy()
        for i in range(self.dim):
            if np.random.rand() < mutation_rate:
                mutation_individual[i] += np.random.uniform(-1, 1)
        return mutation_individual