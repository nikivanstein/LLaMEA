# Description: Adaptive Black Box Optimization Algorithm
# Code: 
# ```python
import random
import numpy as np

class AdaptiveBlackBoxOptimizer:
    """
    An adaptive black box optimization algorithm that adapts its strategy based on the performance of the individual lines of code.
    
    The algorithm uses a combination of random search and perturbation to find the optimal solution, and adapts its strategy based on the performance of the individual lines of code.
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
        self.best_individual = None
        self.best_cost = float('inf')
        self.perturbation_coefficient = 0.3
        self.iteration_coefficient = 0.3
    
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
        
        # Update the best solution and cost if the new solution is better
        if cost < self.best_cost:
            self.best_individual = solution
            self.best_cost = cost
        
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
            
            # Update the best solution and cost if the new solution is better
            if cost < self.best_cost:
                self.best_individual = solution
                self.best_cost = cost
                self.iteration_coefficient = 0.7
            else:
                self.iteration_coefficient = 0.3
        
        return solution, cost

An exception occured: Traceback (most recent call last):
  File "/root/LLaMEA/llamea/llamea.py", line 187, in initialize_single
    new_individual = self.evaluate_fitness(new_individual)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/LLaMEA/mutation_exp.py", line 52, in evaluateBBOB
    algorithm(problem)
  File "<string>", line 42, in __call__
  File "<string>", line 69, in perturb
TypeError: 'NoneType' object is not subscriptable
.

# Description: Adaptive Black Box Optimization Algorithm
# Code: 
# ```python
# ```python
# import random
# import numpy as np

class AdaptiveBlackBoxOptimizer:
    """
    An adaptive black box optimization algorithm that adapts its strategy based on the performance of the individual lines of code.
    
    The algorithm uses a combination of random search and perturbation to find the optimal solution, and adapts its strategy based on the performance of the individual lines of code.
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
        self.best_individual = None
        self.best_cost = float('inf')
        self.perturbation_coefficient = 0.3
        self.iteration_coefficient = 0.3
    
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
        
        # Update the best solution and cost if the new solution is better
        if cost < self.best_cost:
            self.best_individual = solution
            self.best_cost = cost
        
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
            
            # Update the best solution and cost if the new solution is better
            if cost < self.best_cost:
                self.best_individual = solution
                self.best_cost = cost
                self.iteration_coefficient = 0.7
            else:
                self.iteration_coefficient = 0.3
        
        return solution, cost

# Description: Adaptive Black Box Optimization Algorithm
# Code: 
# ```python
# import random
# import numpy as np

class AdaptiveBlackBoxOptimizer:
    """
    An adaptive black box optimization algorithm that adapts its strategy based on the performance of the individual lines of code.
    
    The algorithm uses a combination of random search and perturbation to find the optimal solution, and adapts its strategy based on the performance of the individual lines of code.
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
        self.best_individual = None
        self.best_cost = float('inf')
        self.perturbation_coefficient = 0.3
        self.iteration_coefficient = 0.3
    
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
        
        # Update the best solution and cost if the new solution is better
        if cost < self.best_cost:
            self.best_individual = solution
            self.best_cost = cost
        
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
            
            # Update the best solution and cost if the new solution is better
            if cost < self.best_cost:
                self.best_individual = solution
                self.best_cost = cost
                self.iteration_coefficient = 0.7
            else:
                self.iteration_coefficient = 0.3
        
        return solution, cost

# Usage:
# optimizer = AdaptiveBlackBoxOptimizer(budget=100, dim=10)
# func = lambda x: x**2
# solution, cost = optimizer(func, 1000)
# print(f"Optimal solution: {solution}, Cost: {cost}")