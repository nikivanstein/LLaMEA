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

    def update_solution(self, new_solution, cost):
        """
        Update the solution using a probability of 0.3 to refine its strategy.
        
        Args:
            new_solution (tuple): The new solution.
            cost (float): The cost of the new solution.
        """
        
        # Generate a random perturbation in the search space
        perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Update the solution with the perturbation
        new_solution = (new_solution[0] + perturbation[0], new_solution[1] + perturbation[1])
        
        # Update the solution with the new solution and its cost
        self.update_individual(new_solution, cost)

    def update_individual(self, new_individual, cost):
        """
        Update the individual using a probability of 0.3 to refine its strategy.
        
        Args:
            new_individual (tuple): The new individual.
            cost (float): The cost of the new individual.
        """
        
        # Evaluate the fitness of the individual
        fitness = self.evaluate_fitness(new_individual)
        
        # Update the individual with the new solution and its cost
        self.update_individual(new_individual, fitness, cost)
        
    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of the individual using the given problem.
        
        Args:
            individual (tuple): The individual to evaluate.
        
        Returns:
            float: The fitness of the individual.
        """
        
        # Evaluate the fitness of the individual
        fitness = 0
        
        # Iterate over the search space
        for i in range(self.dim):
            # Evaluate the fitness of the individual at the current position
            fitness += self.search_space[i][0] * individual[i]
        
        return fitness

# Description: Black Box Optimization using Genetic Algorithm with Adaptive Mutation Strategy
# Code: 