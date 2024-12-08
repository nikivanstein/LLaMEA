import random
import numpy as np

class AdaptivePerturbation:
    """
    An adaptive perturbation strategy for the Randomized Black Box Optimization Algorithm.
    
    The algorithm adjusts the perturbation amplitude based on the fitness of the solution.
    """
    
    def __init__(self, budget, dim):
        """
        Initialize the adaptive perturbation strategy.
        
        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.perturbation_amplitude = 0.5
        self.fitness_history = []
    
    def __call__(self, func, num_iterations):
        """
        Optimize the black box function using the adaptive perturbation strategy.
        
        Args:
            func (function): The black box function to optimize.
            num_iterations (int): The number of iterations to run.
        
        Returns:
            tuple: The optimal solution and its cost.
        """
        
        # Initialize the solution and cost
        solution = None
        cost = float('inf')
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            # Evaluate the fitness of the current solution
            fitness = self.evaluate_fitness(func, solution)
            
            # If the fitness is better, update the solution and cost
            if fitness < cost:
                solution = func(solution)
                cost = fitness
            
            # Update the perturbation amplitude based on the fitness
            self.perturbation_amplitude = max(0.1, 0.1 - 0.01 * fitness)
        
        return solution, cost
    
    def evaluate_fitness(self, func, solution):
        """
        Evaluate the fitness of the solution.
        
        Args:
            func (function): The black box function to optimize.
            solution (tuple): The current solution.
        
        Returns:
            float: The fitness of the solution.
        """
        
        # Evaluate the fitness of the solution
        fitness = func(solution)
        
        # Store the fitness history
        self.fitness_history.append(fitness)
        
        # Return the fitness
        return fitness

class RandomizedBlackBoxOptimizer:
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

# Example usage:
def func1(x):
    return x[0]**2 + x[1]**2

optimizer = RandomizedBlackBoxOptimizer(100, 2)
solution, cost = optimizer.func1(np.array([1, 2]))
print(f"Solution: {solution}, Cost: {cost}")

# Update the adaptive perturbation strategy
optimizer = AdaptivePerturbation(100, 2)
solution, cost = optimizer.func1(np.array([1, 2]))
print(f"Solution: {solution}, Cost: {cost}")

# Update the Randomized Black Box Optimization Algorithm
optimizer = RandomizedBlackBoxOptimizer(100, 2)
solution, cost = optimizer.func1(np.array([1, 2]))
print(f"Solution: {solution}, Cost: {cost}")

# Run the optimizer for 100 iterations
optimizer = RandomizedBlackBoxOptimizer(100, 2)
for _ in range(100):
    solution, cost = optimizer.func1(np.array([1, 2]))
    print(f"Solution: {solution}, Cost: {cost}")