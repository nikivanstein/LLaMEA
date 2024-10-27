# Description: Black Box Optimization using MetaHeuristic Algorithm with Refining Strategy
# Code: 
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
    strategy (list): The refining strategy of the algorithm.
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
        self.strategy = None

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
        
        # Apply refining strategy
        self.apply_strategy(opt_solution, opt_cost)
        
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

    def apply_strategy(self, opt_solution, opt_cost):
        """
        Applies the refining strategy to the optimal solution.
        
        Args:
        opt_solution (list): The optimal solution.
        opt_cost (float): The cost of the optimal solution.
        """
        # Define the refining strategy
        if self.strategy == 'line_search':
            # Line search refinement
            for i in range(self.dim):
                new_bounds = [self.bounds[0] + 0.1 * (self.bounds[1] - self.bounds[0]), self.bounds[1] - 0.1 * (self.bounds[1] - self.bounds[0])]
                new_bounds[i] = max(bounds[i], min(new_bounds[i], bounds[i]))
                new_solution = [new_bounds[0] + np.random.uniform(-1, 1) * (new_bounds[1] - new_bounds[0]), new_bounds[1] - 0.1 * (new_bounds[1] - new_bounds[0])]
                new_solution = [max(bounds[i], min(new_solution[i], bounds[i])) for i in range(self.dim)]
                new_solution = [new_bounds[0] + 0.1 * (new_bounds[1] - new_bounds[0]), new_bounds[1] - 0.1 * (new_bounds[1] - new_bounds[0])]
                new_solution[i] = max(bounds[i], min(new_solution[i], bounds[i]))
                new_solution[i] = min(new_solution[i], self.bounds[i])
                new_solution[i] = max(new_solution[i], self.bounds[i])
                new_solution[i] = min(new_solution[i], self.bounds[i])
        elif self.strategy == 'bounded':
            # Bounded refinement
            for i in range(self.dim):
                new_bounds = [self.bounds[0] + np.random.uniform(-1, 1) * (self.bounds[1] - self.bounds[0]), self.bounds[1] - np.random.uniform(1, 0.1)]
                new_bounds[i] = max(bounds[i], min(new_bounds[i], bounds[i]))
                new_solution = [new_bounds[0] + np.random.uniform(-1, 1) * (new_bounds[1] - new_bounds[0]), new_bounds[1] - np.random.uniform(1, 0.1)]
                new_solution[i] = max(bounds[i], min(new_solution[i], bounds[i]))
                new_solution[i] = min(new_solution[i], self.bounds[i])
        elif self.strategy == 'random':
            # Random refinement
            for i in range(self.dim):
                new_bounds = [self.bounds[0] + np.random.uniform(-1, 1) * (self.bounds[1] - self.bounds[0]), self.bounds[1] - np.random.uniform(1, 0.1)]
                new_bounds[i] = max(bounds[i], min(new_bounds[i], bounds[i]))
                new_solution = [new_bounds[0] + np.random.uniform(-1, 1) * (new_bounds[1] - new_bounds[0]), new_bounds[1] - np.random.uniform(1, 0.1)]
                new_solution[i] = max(bounds[i], min(new_solution[i], bounds[i]))
                new_solution[i] = min(new_solution[i], self.bounds[i])
        else:
            raise ValueError("Invalid refining strategy")

# Description: Black Box Optimization using MetaHeuristic Algorithm with Refining Strategy
# Code: 