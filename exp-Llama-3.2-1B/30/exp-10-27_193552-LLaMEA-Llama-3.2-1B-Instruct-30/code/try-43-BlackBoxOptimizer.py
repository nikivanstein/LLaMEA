import random
import numpy as np

class BlackBoxOptimizer:
    """
    A metaheuristic algorithm to optimize black box functions.
    
    The algorithm uses a combination of random search, perturbation, and evolutionary strategies to find the optimal solution.
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
        population = [self.evaluate_fitness(func, solution) for solution in self.evaluate_fitness(func, [random.uniform(-5.0, 5.0)] * self.dim)]
        
        # Evolve the population over the specified number of iterations
        for _ in range(num_iterations):
            # Select the fittest solutions
            fittest_solutions = [solution for solution, _ in zip(population, range(num_iterations))]
            
            # Create a new population by perturbing the fittest solutions
            new_population = []
            for _ in range(len(fittest_solutions)):
                # Perturb a solution
                perturbed_solution = self.perturb(fittest_solutions[_])
                
                # Evaluate the new solution
                new_cost = func(perturbed_solution)
                
                # Add the new solution to the new population
                new_population.append(perturbed_solution)
            
            # Update the population with the new population
            population = new_population
    
    def evaluate_fitness(self, func, individual):
        """
        Evaluate the fitness of an individual.
        
        Args:
            func (function): The black box function.
            individual (tuple): The individual to evaluate.
        
        Returns:
            float: The fitness of the individual.
        """
        
        # Evaluate the individual using the black box function
        fitness = func(individual)
        
        # Update the fitness of the individual
        individual = (individual[0] + random.uniform(-1, 1), individual[1] + random.uniform(-1, 1))
        
        return fitness
    
# Description: Novel Randomized Black Box Optimization Algorithm using Perturbation and Evolutionary Strategies
# Code: 