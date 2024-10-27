import numpy as np
from scipy.optimize import minimize
import random

class MetaHeuristic:
    """
    A metaheuristic algorithm for solving black box optimization problems.
    
    The algorithm uses a combination of local search and gradient-based optimization to find the optimal solution.
    The strategy is tuned using a meta-learning approach to adapt to different problems.
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
        self.tuning_strategy = None

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

    def tune_strategy(self, func, bounds, initial_strategy, num_tunes):
        """
        Tunes the strategy using a meta-learning approach.
        
        Args:
        func (function): The black box function to optimize.
        bounds (list): The current bounds of the search space.
        initial_strategy (list): The initial strategy.
        num_tunes (int): The number of tunes to perform.
        
        Returns:
        list: The tuned strategy.
        """
        # Initialize the tuned strategy
        tuned_strategy = initial_strategy
        
        # Perform the specified number of tunes
        for _ in range(num_tunes):
            # Evaluate the current strategy
            current_cost = self.evaluate_fitness(tuned_strategy, func, bounds)
            
            # Tune the strategy
            for i in range(self.dim):
                # Generate a new strategy by perturbing the current strategy
                new_strategy = [self.bounds[0] + np.random.uniform(-1, 1) * (self.bounds[1] - self.bounds[0]), self.bounds[1] + np.random.uniform(-1, 1) * (self.bounds[0] - self.bounds[1])]
                
                # Evaluate the new strategy
                new_cost = self.evaluate_fitness(new_strategy, func, bounds)
                
                # Update the tuned strategy if necessary
                if new_cost < current_cost:
                    tuned_strategy = new_strategy
        
        # Return the tuned strategy
        return tuned_strategy

# Description: Black Box Optimization using MetaHeuristic Algorithm with Tuned Strategy
# Code: 