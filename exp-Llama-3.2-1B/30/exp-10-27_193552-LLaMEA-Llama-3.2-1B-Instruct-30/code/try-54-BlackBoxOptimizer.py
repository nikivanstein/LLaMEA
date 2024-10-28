import random
import numpy as np
from scipy.optimize import minimize

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

def bbo_bound(func, lower, upper):
    """
    Define the black box function to optimize.
    
    Args:
        func (function): The black box function to optimize.
        lower (float): The lower bound of the search space.
        upper (float): The upper bound of the search space.
    """
    return lambda x: func(x[0], x[1])

def bbo_optimize(func, lower, upper, budget, dim):
    """
    Optimize the black box function using the randomized black box optimization algorithm.
    
    Args:
        func (function): The black box function to optimize.
        lower (float): The lower bound of the search space.
        upper (float): The upper bound of the search space.
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
    """
    
    # Define the search space
    search_space = [bbo_bound(func, lower, upper) for _ in range(dim)]
    
    # Initialize the population
    population = [random.uniform(lower, upper) for _ in range(100)]
    
    # Run the optimizer for the specified number of iterations
    for _ in range(100):
        # Select the fittest individual
        fittest = population[np.argmax([self.evaluate_fitness(individual) for individual in population])]
        
        # Mutate the fittest individual
        mutated = fittest[:dim] + [self.perturb(individual) for individual in population]
        
        # Evaluate the fitness of the mutated individual
        fitness = self.evaluate_fitness(mutated)
        
        # Replace the fittest individual with the mutated individual
        population[np.argmax([self.evaluate_fitness(individual) for individual in population])] = mutated
        
        # If the optimizer has reached the budget, break the loop
        if self.evaluate_fitness(population[np.argmax([self.evaluate_fitness(individual) for individual in population])]) < fitness:
            break
    
    # Return the fittest individual and its fitness
    return population[np.argmax([self.evaluate_fitness(individual) for individual in population])], self.evaluate_fitness(population[np.argmax([self.evaluate_fitness(individual) for individual in population])])

def evaluate_fitness(individual):
    """
    Evaluate the fitness of an individual.
    
    Args:
        individual (tuple): The individual to evaluate.
    
    Returns:
        float: The fitness of the individual.
    """
    
    # Evaluate the function at the individual
    func = lambda x: x[0]**2 + x[1]**2
    return func(individual)

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

optimizer = BlackBoxOptimizer(100, 2)
solution, cost = optimizer(func, -5.0, 5.0, 100, 2)
print(f"Optimal solution: {solution}, Cost: {cost}")