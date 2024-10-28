# Description: Randomized Black Box Optimization Algorithm with Adaptive Perturbation Strategy
# Code: 
# ```python
import random
import numpy as np

class BlackBoxOptimizer:
    """
    A metaheuristic algorithm to optimize black box functions.
    
    The algorithm uses a combination of random search and perturbation to find the optimal solution.
    It handles a wide range of tasks and can be adapted to various optimization problems.
    """
    
    def __init__(self, budget, dim):
        """
        Initialize the optimizer with a budget and dimensionality.
        
        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = [(-5.0, 5.0)] * dim
        self.func_evaluations = 0
        
        # Initialize the perturbation strategy
        self.perturbation_strategy = 'random'
        
    def __call__(self, func):
        """
        Optimize the black box function using the optimizer.
        
        Args:
            func (function): The black box function to optimize.
        
        Returns:
            tuple: The optimal solution and its cost.
        """
        
        # Initialize the solution and cost
        solution = None
        cost = float('inf')
        
        # Perform random search
        for _ in range(self.budget):
            # Perturb the current solution
            if self.perturbation_strategy == 'random':
                perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
            elif self.perturbation_strategy == 'adaptive':
                perturbation = (random.uniform(-1, 1) / self.dim, random.uniform(-1, 1) / self.dim)
            else:
                raise ValueError("Invalid perturbation strategy")
            
            # Evaluate the new solution
            new_cost = func(self.evaluate_individual(perturbation))
            
            # Update the solution and cost if the new solution is better
            if new_cost < cost:
                solution = self.evaluate_individual(perturbation)
                cost = new_cost
        
        return solution, cost
    
    def perturb(self, individual):
        """
        Perturb the current individual.
        
        Args:
            individual (tuple): The current individual.
        
        Returns:
            tuple: The perturbed individual.
        """
        
        # Generate a random perturbation in the search space
        if self.perturbation_strategy == 'random':
            perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
        elif self.perturbation_strategy == 'adaptive':
            perturbation = (random.uniform(-1 / self.dim, 1 / self.dim), random.uniform(-1 / self.dim, 1 / self.dim))
        else:
            raise ValueError("Invalid perturbation strategy")
        
        # Update the individual with the perturbation
        return (individual[0] + perturbation[0], individual[1] + perturbation[1])
    
    def evaluate_individual(self, perturbation):
        """
        Evaluate the individual with the given perturbation.
        
        Args:
            perturbation (tuple): The perturbation to apply.
        
        Returns:
            tuple: The evaluated individual.
        """
        
        # Evaluate the individual without the perturbation
        evaluated_individual = self.evaluate_individual(perturbation[0])
        
        # Update the individual with the perturbation
        return (evaluated_individual[0] + perturbation[0], evaluated_individual[1] + perturbation[1])
    
    def run(self, func, num_iterations):
        """
        Run the optimizer for a specified number of iterations.
        
        Args:
            func (function): The black box function to optimize.
            num_iterations (int): The number of iterations to run.
        
        Returns:
            tuple: The optimal solution and its cost.
        """
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            solution, cost = self(func)
            self.func_evaluations += 1
            
            # If the optimizer has reached the budget, break the loop
            if self.func_evaluations >= self.budget:
                break
        
        return solution, cost