import numpy as np
from scipy.optimize import minimize

class MetaHeuristic:
    """
    A metaheuristic algorithm for solving black box optimization problems.
    
    The algorithm uses a combination of local search and gradient-based optimization to find the optimal solution.
    
    Attributes:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    func (function): The black box function to optimize.
    search_space (list): The range of the search space.
    bounds (list): The bounds of the search space.
    learning_rate (float): The learning rate for the adaptive learning rate strategy.
    """

    def __init__(self, budget, dim):
        """
        Initializes the MetaHeuristic algorithm.
        
        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.bounds = None
        self.learning_rate = 0.1

    def __call__(self, func):
        """
        Optimizes the black box function using MetaHeuristic.
        
        Args:
        func (function): The black box function to optimize.
        
        Returns:
        tuple: A tuple containing the optimal solution and its cost.
        """
        if self.func is None:
            raise ValueError("The black box function must be initialized before calling this method.")
        
        # Initialize the search space
        self.search_space = [self.bounds] * self.dim
        self.bounds = [(-5.0, 5.0)] * self.dim
        
        # Initialize the optimal solution and its cost
        opt_solution = None
        opt_cost = float('inf')
        
        # Perform local search
        for _ in range(self.budget):
            # Generate a new solution by perturbing the current solution
            new_solution = self.perturb(self.search_space, self.bounds)
            
            # Evaluate the new solution using the black box function
            new_cost = self.func(new_solution)
            
            # Update the optimal solution and its cost if necessary
            if new_cost < opt_cost:
                opt_solution = new_solution
                opt_cost = new_cost
            
            # Update the learning rate based on the fitness
            self.learning_rate = 1 / (1 + self.learning_rate * np.abs(self.func(opt_solution)))
        
        # Return the optimal solution and its cost
        return opt_solution, opt_cost

    def perturb(self, search_space, bounds):
        """
        Generates a new solution by perturbing the current solution.
        
        Args:
        search_space (list): The current search space.
        bounds (list): The current bounds of the search space.
        
        Returns:
        list: A new solution generated by perturbing the current solution.
        """
        # Generate a new solution by randomly perturbing the current solution
        new_solution = [self.bounds[0] + np.random.uniform(-1, 1) * (self.bounds[1] - self.bounds[0]) for _ in range(self.dim)]
        
        # Ensure the new solution is within the bounds
        new_solution = [max(bounds[i], min(new_solution[i], bounds[i])) for i in range(self.dim)]
        
        return new_solution

# Description: MetaHeuristic with Adaptive Learning Rate
# Code: 
# ```python
# MetaHeuristic with Adaptive Learning Rate
# Code: 