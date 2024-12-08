# Description: Randomized Black Box Optimization Algorithm (RBBOA)
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
        
        return solution, cost

def evaluate_bbof(func, budget, dim):
    """
    Evaluate the black box function for a specified number of iterations.
    
    Args:
        func (function): The black box function to optimize.
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
    
    Returns:
        tuple: The optimal solution and its cost.
    """
    
    # Initialize the optimizer
    optimizer = BlackBoxOptimizer(budget, dim)
    
    # Run the optimizer for the specified number of iterations
    solution, cost = optimizer(func, 1000)
    
    # Evaluate the fitness of the solution
    fitness = func(solution)
    
    return optimizer, solution, cost, fitness

def perturb_solution(solution, optimizer):
    """
    Perturb the current solution using a probability of 0.3.
    
    Args:
        solution (tuple): The current solution.
        optimizer (BlackBoxOptimizer): The optimizer to use.
    
    Returns:
        tuple: The perturbed solution.
    """
    
    # Generate a random perturbation in the search space
    perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
    
    # Update the solution with the perturbation
    solution = (solution[0] + perturbation[0], solution[1] + perturbation[1])
    
    # Apply a probability of 0.3 to refine the strategy
    if random.random() < 0.3:
        solution = perturb_solution(solution, optimizer)
    
    return solution

# Example usage:
def black_box_function(x):
    return x**2 + 2*x + 1

optimizer, solution, cost, fitness = evaluate_bbof(black_box_function, 1000, 2)

print(f"Optimal solution: {solution}")
print(f"Cost: {cost}")
print(f"Fitness: {fitness}")

perturbed_solution = perturb_solution(solution, optimizer)
print(f"Perturbed solution: {perturbed_solution}")