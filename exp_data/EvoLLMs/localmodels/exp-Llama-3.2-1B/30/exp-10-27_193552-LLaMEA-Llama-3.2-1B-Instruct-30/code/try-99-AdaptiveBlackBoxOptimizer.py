# Description: A novel metaheuristic algorithm to optimize black box functions using a combination of random search, perturbation, and adaptive line search.
# Code: 
# ```python
import random
import numpy as np

class AdaptiveBlackBoxOptimizer:
    """
    A metaheuristic algorithm to optimize black box functions using a combination of random search, perturbation, and adaptive line search.
    
    The algorithm uses adaptive line search to improve the convergence rate and adapt to the search space.
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
        self.population = None
        self.population_history = None
        self.line_search = False

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
        
        # Initialize the population with random solutions
        self.population = self.initialize_population(func, self.budget, self.dim)
        
        # Perform random search
        for _ in range(self.budget):
            # Perturb the current solution
            perturbed_solution = self.perturb(self.population)
            
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
    
    def initialize_population(self, func, budget, dim):
        """
        Initialize the population with random solutions.
        
        Args:
            func (function): The black box function to optimize.
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        
        Returns:
            list: The initialized population.
        """
        
        # Initialize the population with random solutions
        population = [func(np.random.rand(dim)) for _ in range(budget)]
        
        return population
    
    def adaptive_line_search(self, solution, func, step_size, max_iter):
        """
        Perform adaptive line search to improve the convergence rate.
        
        Args:
            solution (tuple): The current solution.
            func (function): The black box function to optimize.
            step_size (float): The step size to update the solution.
            max_iter (int): The maximum number of iterations to perform.
        
        Returns:
            tuple: The updated solution and its cost.
        """
        
        # Initialize the updated solution
        updated_solution = solution
        
        # Perform adaptive line search
        for _ in range(max_iter):
            # Evaluate the function at the current solution
            new_cost = func(updated_solution)
            
            # Update the solution using the adaptive line search
            updated_solution = (updated_solution[0] + step_size * (updated_solution[1] - solution[1]), updated_solution[1] + step_size * (updated_solution[0] - solution[0]))
            
            # Check if the solution has converged
            if np.abs(updated_solution[0] - updated_solution[1]) < 1e-6:
                break
        
        return updated_solution, new_cost
    
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
        
        # Perform adaptive line search
        self.line_search = True
        solution, cost = self.adaptive_line_search(solution, func, 0.1, 100)
        
        # Update the population with the adaptive line search solution
        self.population = [func(np.random.rand(self.dim)) for _ in range(self.budget)]
        
        # Update the population history
        self.population_history = self.population_history + [self.population]
        
        return solution, cost

# Example usage:
# optimizer = AdaptiveBlackBoxOptimizer(100, 10)
# solution, cost = optimizer(func, 1000)
# print("Optimal solution:", solution)
# print("Cost:", cost)