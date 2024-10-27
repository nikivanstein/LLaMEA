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
    adaptive_perturbation (list): The adaptive perturbation strategy.
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
        self.adaptive_perturbation = None

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
            # Initialize the adaptive perturbation strategy
            adaptive_perturbation_strategy = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            # Generate a new solution by perturbing the current solution
            new_solution = self.perturb(self.search_space, self.bounds, adaptive_perturbation_strategy)
            
            # Evaluate the new solution using the black box function
            new_cost = self.func(new_solution)
            
            # Update the optimal solution and its cost if necessary
            if new_cost < opt_cost:
                opt_solution = new_solution
                opt_cost = new_cost
        
        # Return the optimal solution and its cost
        return opt_solution, opt_cost

    def perturb(self, search_space, bounds, adaptive_perturbation_strategy):
        """
        Generates a new solution by perturbing the current solution.
        
        Args:
        search_space (list): The current search space.
        bounds (list): The current bounds of the search space.
        adaptive_perturbation_strategy (list): The adaptive perturbation strategy.
        
        Returns:
        list: A new solution generated by perturbing the current solution.
        """
        # Initialize the new solution
        new_solution = [self.bounds[0] + np.random.uniform(-1, 1) * (self.bounds[1] - self.bounds[0]) for _ in range(self.dim)]
        
        # Apply the adaptive perturbation strategy
        for i, strategy in enumerate(adaptive_perturbation_strategy):
            new_solution = [max(bounds[i], min(new_solution[i], bounds[i])) for i in range(self.dim)]
        
        return new_solution