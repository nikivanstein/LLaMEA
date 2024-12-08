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
    
    def mutate(self, solution):
        """
        Mutate the current solution.
        
        Args:
            solution (tuple): The current solution.
        
        Returns:
            tuple: The mutated solution.
        """
        
        # Generate a random perturbation in the search space
        perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Update the solution with the perturbation
        solution = (solution[0] + perturbation[0], solution[1] + perturbation[1])
        
        return solution
    
    def evaluate_fitness(self, solution):
        """
        Evaluate the fitness of the current solution.
        
        Args:
            solution (tuple): The current solution.
        
        Returns:
            float: The fitness of the solution.
        """
        
        # Evaluate the fitness of the current solution
        fitness = func(solution)
        
        # Refine the solution based on the fitness
        refined_solution = self.refine(solution, fitness)
        
        return refined_solution, fitness
    
    def refine(self, solution, fitness):
        """
        Refine the solution based on the fitness.
        
        Args:
            solution (tuple): The current solution.
            fitness (float): The fitness of the solution.
        
        Returns:
            tuple: The refined solution.
        """
        
        # Calculate the probability of refinement based on the fitness
        probability = 0.3 * fitness / self.func_evaluations
        
        # Refine the solution based on the probability
        if random.random() < probability:
            return self.mutate(solution)
        else:
            return solution
    
    def select(self, solution, fitness):
        """
        Select the next solution based on the fitness.
        
        Args:
            solution (tuple): The current solution.
            fitness (float): The fitness of the solution.
        
        Returns:
            tuple: The selected solution.
        """
        
        # Select the next solution based on the fitness
        selected_solution = (solution[0] + fitness / self.func_evaluations, solution[1] + fitness / self.func_evaluations)
        
        return selected_solution
    
    def __str__(self):
        """
        Return a string representation of the optimizer.
        
        Returns:
            str: A string representation of the optimizer.
        """
        
        return "Randomized Black Box Optimization Algorithm with Refining Strategy"