import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveBBOBOptimizer:
    """
    An adaptive differential evolution metaheuristic algorithm for solving black box optimization problems.
    
    The algorithm uses differential evolution to search for the optimal solution in the search space.
    It adapts its strategy based on the performance of the individual lines of the selected solution.
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
        self.fitness_history = []

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
        
        # Calculate the fitness of the optimal solution
        fitness = -res.fun
        
        # Store the fitness history
        self.fitness_history.append(fitness)
        
        # If the number of evaluations exceeds the budget, refine the strategy
        if len(self.fitness_history) >= self.budget:
            # Calculate the average fitness
            avg_fitness = np.mean(self.fitness_history)
            
            # If the average fitness is below 0.45, increase the number of evaluations
            if avg_fitness < 0.45:
                self.budget *= 2
            # Otherwise, decrease the number of evaluations
            else:
                self.budget /= 2
            
            # Refine the strategy by changing the bounds of the individual lines
            x = np.linspace(-5.0, 5.0, self.dim)
            y = func(x)
            res = differential_evolution(lambda x: -y, [(x, y)], x0=x, bounds=((None, None), (None, None)), n_iter=self.budget)
            self.fitness_history.append(-res.fun)
        
        # Return the optimal solution and the corresponding objective value
        return res.x, -res.fun


# Example usage:
optimizer = AdaptiveBBOBOptimizer(100, 10)
func = lambda x: x**2
optimal_solution, optimal_fitness = optimizer(func)
print("Optimal solution:", optimal_solution)
print("Optimal fitness:", optimal_fitness)