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
        self.change_probability = 0.3  # Adapt the probability of change based on the function's area over convergence curve (AOCC)

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
        
        # Update the probability of change based on the function's area over convergence curve (AOCC)
        self.change_probability = max(0.01, 1 - (self.func_evaluations / self.budget))
        
        # Refine the solution based on the updated probability of change
        refined_solution = self.refine_solution(solution, self.change_probability)
        
        # If the optimizer has reached the budget, break the loop
        if self.func_evaluations >= self.budget:
            break
        
        return refined_solution, cost

    def refine_solution(self, solution, change_probability):
        """
        Refine the solution based on the updated probability of change.
        
        Args:
            solution (tuple): The current solution.
            change_probability (float): The updated probability of change.
        
        Returns:
            tuple: The refined solution.
        """
        
        # Initialize the population with the refined solution
        population = [solution]
        
        # Perform random search with the updated probability of change
        for _ in range(self.budget):
            # Perturb the current solution
            perturbed_solution = self.perturb(population[-1])
            
            # Evaluate the new solution
            new_cost = func(perturbed_solution)
            
            # Update the population and cost if the new solution is better
            if new_cost < self.func_evaluations:
                population.append(perturbed_solution)
                self.func_evaluations += 1
        
        # Refine the population based on the updated probability of change
        refined_solution = self.refine_population(population, change_probability)
        
        return refined_solution
    
    def refine_population(self, population, change_probability):
        """
        Refine the population based on the updated probability of change.
        
        Args:
            population (list): The population.
            change_probability (float): The updated probability of change.
        
        Returns:
            list: The refined population.
        """
        
        # Initialize the population with the refined solution
        refined_population = population
        
        # Perform random search with the updated probability of change
        for _ in range(self.budget):
            # Perturb the current solution
            perturbed_solution = self.perturb(refined_population[-1])
            
            # Evaluate the new solution
            new_cost = func(perturbed_solution)
            
            # Update the population and cost if the new solution is better
            if new_cost < self.func_evaluations:
                refined_population.append(perturbed_solution)
                self.func_evaluations += 1
        
        # Refine the population based on the updated probability of change
        refined_population = self.refine_population(refined_population, change_probability)
        
        return refined_population