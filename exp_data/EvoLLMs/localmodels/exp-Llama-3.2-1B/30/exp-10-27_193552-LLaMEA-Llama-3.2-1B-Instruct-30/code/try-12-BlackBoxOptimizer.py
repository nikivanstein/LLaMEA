# Description: Black Box Optimization using Genetic Algorithm with Refinement
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    """
    A metaheuristic algorithm to optimize black box functions.
    
    The algorithm uses a combination of random search, perturbation, and refinement to find the optimal solution.
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
        self.refinement = 0
    
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
            perturbed_solution = self.perturb(solution)
            
            # Evaluate the new solution
            new_cost = func(perturbed_solution)
            
            # Update the solution and cost if the new solution is better
            if new_cost < cost:
                solution = perturbed_solution
                cost = new_cost
        
        return solution, cost
    
    def perturb(self, solution):
        """
        Perturb the current solution.
        
        Args:
            solution (tuple): The current solution.
        
        Returns:
            tuple: The perturbed solution.
        """
        
        # Generate a random perturbation in the search space
        perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Update the solution with the perturbation
        solution = (solution[0] + perturbation[0], solution[1] + perturbation[1])
        
        return solution
    
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
            
            # Refine the solution based on the probability of the current solution
            self.refine(solution)
        
        return solution, cost
    
    def refine(self, solution):
        """
        Refine the solution based on the probability of the current solution.
        
        Args:
            solution (tuple): The current solution.
        """
        
        # Generate a new solution based on the probability of the current solution
        new_solution = (solution[0] + random.uniform(-1, 1) / 10, solution[1] + random.uniform(-1, 1) / 10)
        
        # Evaluate the new solution
        new_cost = func(new_solution)
        
        # Update the solution and cost if the new solution is better
        if new_cost < float('inf'):
            solution = new_solution
            cost = new_cost
        
        # Update the refinement probability
        self.refinement = 0.3 - (1 - self.refinement) * 0.01
    
    def evaluate_fitness(self, individual, logger):
        """
        Evaluate the fitness of an individual.
        
        Args:
            individual (tuple): The individual to evaluate.
            logger (object): The logger to use.
        
        Returns:
            float: The fitness of the individual.
        """
        
        # Evaluate the fitness of the individual
        updated_individual = self.f(individual, logger)
        
        # Log the fitness of the updated individual
        logger.info(f'Fitness of updated individual: {updated_individual}')
        
        return updated_individual
    
    def f(self, individual, logger):
        """
        Evaluate the fitness of an individual using the Black Box Optimization algorithm.
        
        Args:
            individual (tuple): The individual to evaluate.
            logger (object): The logger to use.
        
        Returns:
            float: The fitness of the individual.
        """
        
        # Evaluate the fitness of the individual
        updated_individual = self.evaluate_fitness(individual, logger)
        
        # Log the fitness of the updated individual
        logger.info(f'Fitness of updated individual: {updated_individual}')
        
        return updated_individual