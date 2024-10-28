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
        
        # Initialize the population with random solutions
        population = [self.evaluate_fitness(solution) for solution in range(self.budget)]
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            # Select the fittest individuals
            fittest_individuals = sorted(population, key=self.evaluate_fitness, reverse=True)[:self.budget//2]
            
            # Create a new population by perturbing the fittest individuals
            new_population = [self.evaluate_fitness(solution) for solution in fittest_individuals]
            for i in range(self.budget//2):
                new_population[i] = self.perturb(new_population[i])
            
            # Update the population
            population = new_population
        
        return population[0], population[-1]

    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual.
        
        Args:
            individual (tuple): The individual to evaluate.
        
        Returns:
            int: The fitness of the individual.
        """
        
        # Evaluate the fitness of the individual
        fitness = func(individual)
        
        # Update the fitness of the individual
        individual = (individual[0] + fitness, individual[1])
        
        return fitness

# Description: Black Box Optimization Algorithm using Evolutionary Strategy
# Code: 