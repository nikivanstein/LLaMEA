import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import expit

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
        
        # Return the optimal solution and the corresponding objective value
        return res.x, -res.fun

    def adapt_strategy(self, res):
        """
        Adapt the strategy of the algorithm based on the optimization result.
        
        Args:
            res (tuple): The optimization result.
        """
        # Calculate the fitness of the individual
        fitness = -res.fun
        
        # Calculate the probability of changing the individual
        prob_change = 1 - expit(fitness / 10)
        
        # Update the individual with the new strategy
        new_individual = res.x
        if np.random.rand() < prob_change:
            new_individual = np.random.uniform(-5.0, 5.0, self.dim)
        
        # Evaluate the new individual
        new_fitness = -np.min([func(new_individual) for func in self.f])
        
        # Update the fitness of the individual
        self.f[new_individual] = new_fitness

    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual.
        
        Args:
            individual (tuple): The individual to evaluate.
        
        Returns:
            float: The fitness of the individual.
        """
        # Create a grid of points in the search space
        x = np.linspace(-5.0, 5.0, self.dim)
        
        # Evaluate the black box function at each point
        y = func(x)
        
        # Perform the optimization using differential evolution
        res = differential_evolution(lambda x: -y, [(x, y)], x0=individual, bounds=((None, None), (None, None)), n_iter=1)
        
        # Return the fitness of the individual
        return res.fun