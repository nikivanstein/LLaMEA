# Description: Adaptive Random Search with Perturbation Optimization Algorithm
# Code: 
# ```python
import random
import numpy as np

class AdaptiveRandomSearchWithPerturbationOptimizer:
    """
    An adaptive random search with perturbation optimization algorithm.
    
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
        self.perturbation_factor = 0.3
        self.perturbation_threshold = 0.1
        self.perturbation_strategy = "random"
    
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
        
        # Run adaptive random search
        for _ in range(self.budget):
            # Evaluate the current solution
            new_cost = func(solution)
            
            # If the solution is not optimal, perturb it
            if new_cost > cost:
                if self.perturbation_strategy == "random":
                    # Perturb the current solution
                    perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
                    solution = (solution[0] + perturbation[0], solution[1] + perturbation[1])
                elif self.perturbation_strategy == "adaptive":
                    # Apply the perturbation strategy
                    if random.random() < self.perturbation_factor:
                        solution = self.perturb(solution)
                elif self.perturbation_strategy == "threshold":
                    # Use a threshold-based perturbation strategy
                    if random.random() < self.perturbation_threshold:
                        solution = self.perturb(solution)
            
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
        
        # Generate a perturbation in the search space
        if self.perturbation_strategy == "random":
            perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
        elif self.perturbation_strategy == "adaptive":
            if random.random() < self.perturbation_factor:
                perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
        elif self.perturbation_strategy == "threshold":
            if random.random() < self.perturbation_threshold:
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
        
        return solution, cost

# Description: Adaptive Random Search with Perturbation Optimization Algorithm
# Code: 
# ```python
# Adaptive Random Search with Perturbation Optimization Algorithm
# ```
# ```python
def perturb_bbof(func, perturbation, search_space):
    """
    Perturb the current solution using a perturbation strategy.
    
    Args:
        func (function): The black box function to optimize.
        perturbation (tuple): The perturbation to apply.
        search_space (list): The search space.
    
    Returns:
        tuple: The perturbed solution.
    """
    
    # Generate a perturbation in the search space
    perturbation = perturbation
    
    # Update the solution with the perturbation
    solution = (search_space[0] + perturbation[0], search_space[1] + perturbation[1])
    
    return solution

# Code for the Adaptive Random Search with Perturbation Optimization Algorithm
# ```python
# Adaptive Random Search with Perturbation Optimization Algorithm
# ```
# ```python
optimizer = AdaptiveRandomSearchWithPerturbationOptimizer(budget=100, dim=10)
# Define the BBOF function to optimize
def bbof_func(x):
    return x[0]**2 + x[1]**2

# Optimize the BBOF function using the optimizer
solution, cost = optimizer(bbof_func, bbof_func, [(-5.0, 5.0)] * 10)
# Print the optimal solution and its cost
print("Optimal Solution:", solution)
print("Optimal Cost:", cost)