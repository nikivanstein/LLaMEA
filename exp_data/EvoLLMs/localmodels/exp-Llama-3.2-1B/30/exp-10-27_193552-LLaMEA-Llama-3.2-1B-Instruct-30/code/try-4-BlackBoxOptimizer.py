import random
import operator
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

def perturb_solution(solution, perturbation):
    """
    Perturb a solution using a random perturbation.
    
    Args:
        solution (tuple): The current solution.
        perturbation (tuple): The random perturbation.
    
    Returns:
        tuple: The perturbed solution.
    """
    
    # Generate a random perturbation in the search space
    perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
    
    # Update the solution with the perturbation
    solution = (solution[0] + perturbation[0], solution[1] + perturbation[1])
    
    return solution

def mutation_exp(individual, self, num_iterations):
    """
    Perform genetic mutation on an individual.
    
    Args:
        individual (tuple): The current individual.
        self (BlackBoxOptimizer): The optimizer instance.
        num_iterations (int): The number of iterations to perform.
    
    Returns:
        tuple: The mutated individual.
    """
    
    # Perform genetic mutation
    for _ in range(num_iterations):
        # Select a random individual from the population
        parent1, parent2 = random.sample([individual, individual], 2)
        
        # Perform crossover
        child = (parent1[0] + 0.5 * (parent1[1] + parent2[1]), parent1[1] + 0.5 * (parent1[0] + parent2[0]))
        
        # Perform mutation
        child = perturb_solution(child, (random.uniform(-1, 1), random.uniform(-1, 1)))
        
        # Replace the parent with the child
        individual = child
    
    return individual

def fitness_exp(individual, func):
    """
    Evaluate the fitness of an individual.
    
    Args:
        individual (tuple): The current individual.
        func (function): The black box function to evaluate.
    
    Returns:
        float: The fitness of the individual.
    """
    
    # Evaluate the fitness of the individual
    return func(individual)

# Create a new optimizer instance
optimizer = BlackBoxOptimizer(100, 10)

# Define a function to optimize
def func(x):
    return x[0]**2 + x[1]**2

# Run the optimizer for 1000 iterations
solution, cost = optimizer.run(func, 1000)

# Print the result
print("Optimal solution:", solution)
print("Cost:", cost)

# Update the optimizer instance
optimizer = BlackBoxOptimizer(100, 10)

# Define a function to optimize
def func(x):
    return x[0]**2 + x[1]**2 + x[2]**2

# Run the optimizer for 1000 iterations
solution, cost = optimizer.run(func, 1000)

# Print the result
print("Optimal solution:", solution)
print("Cost:", cost)

# Update the optimizer instance
optimizer = BlackBoxOptimizer(100, 10)

# Define a function to optimize
def func(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2

# Run the optimizer for 1000 iterations
solution, cost = optimizer.run(func, 1000)

# Print the result
print("Optimal solution:", solution)
print("Cost:", cost)

# Update the optimizer instance
optimizer = BlackBoxOptimizer(100, 10)

# Define a function to optimize
def func(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2

# Run the optimizer for 1000 iterations
solution, cost = optimizer.run(func, 1000)

# Print the result
print("Optimal solution:", solution)
print("Cost:", cost)

# Update the optimizer instance
optimizer = BlackBoxOptimizer(100, 10)

# Define a function to optimize
def func(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2

# Run the optimizer for 1000 iterations
solution, cost = optimizer.run(func, 1000)

# Print the result
print("Optimal solution:", solution)
print("Cost:", cost)