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
        self.population = None
        self.population_history = None

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

# Example usage:
# optimizer = BlackBoxOptimizer(100, 10)
# solution, cost = optimizer(func, 1000)
# print("Optimal solution:", solution)
# print("Cost:", cost)

# Refining the strategy for a specific task
def task1_optimizer(budget, dim):
    """
    A refined version of the BlackBoxOptimizer for a specific task.
    
    The algorithm uses a combination of random search and perturbation with a focus on adaptability to different tasks.
    """
    
    # Initialize the population with random solutions
    population = self.initialize_population(task1_func, budget, dim)
    
    # Perform random search
    for _ in range(budget):
        # Perturb the current solution
        perturbed_solution = self.perturb(population)
        
        # Evaluate the new solution
        new_cost = task1_func(perturbed_solution)
        
        # Update the solution and cost if the new solution is better
        if new_cost < self.func_evaluations:
            population = perturbed_solution
    
    return population

# Define the task1_func for the task1_optimizer
def task1_func(solution):
    """
    The task1_func to optimize.
    
    The function is defined as follows:
    f(x) = x^2 + 2x + 1
    """
    
    return solution[0]**2 + 2*solution[0] + 1

# Run the task1_optimizer for a specified number of iterations
task1_solution, task1_cost = task1_optimizer(100, 10)
print("Optimal solution for task1:", task1_solution)
print("Cost for task1:", task1_cost)