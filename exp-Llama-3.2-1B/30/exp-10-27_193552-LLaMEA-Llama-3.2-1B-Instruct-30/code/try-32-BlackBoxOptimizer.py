# Description: Randomized Black Box Optimization Algorithm
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
        
        # Initialize the population with random solutions
        population = self.initialize_population(num_iterations)
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            # Select the fittest individual
            fittest_individual = self.select_fittest(population)
            
            # Perform mutation on the fittest individual
            mutated_individual = self.mutate(fittest_individual)
            
            # Evaluate the mutated individual
            new_cost = func(mutated_individual)
            
            # Update the fittest individual if the mutated individual is better
            if new_cost < self.func_evaluations:
                self.func_evaluations = new_cost
                fittest_individual = mutated_individual
        
        # Return the fittest individual and its cost
        return fittest_individual, self.func_evaluations
    
    def initialize_population(self, num_iterations):
        """
        Initialize the population with random solutions.
        
        Args:
            num_iterations (int): The number of iterations to run.
        
        Returns:
            list: The population of individuals.
        """
        
        # Initialize the population with random solutions
        population = []
        for _ in range(num_iterations):
            # Generate a random solution
            solution = (random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))
            population.append(solution)
        
        return population
    
    def select_fittest(self, population):
        """
        Select the fittest individual from the population.
        
        Args:
            population (list): The population of individuals.
        
        Returns:
            tuple: The fittest individual.
        """
        
        # Select the fittest individual based on the fitness function
        fittest_individual = population[0]
        for individual in population:
            fitness = self.func(individual)
            if fitness < self.func_evaluations:
                fittest_individual = individual
                self.func_evaluations = fitness
        
        return fittest_individual
    
    def mutate(self, individual):
        """
        Mutate the individual.
        
        Args:
            individual (tuple): The individual to mutate.
        
        Returns:
            tuple: The mutated individual.
        """
        
        # Generate a random mutation in the search space
        mutation = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Update the individual with the mutation
        individual = (individual[0] + mutation[0], individual[1] + mutation[1])
        
        return individual
    
# Description: Randomized Black Box Optimization Algorithm
# Code: 
# ```python
def evaluate_fitness(individual, func):
    """
    Evaluate the fitness of an individual.
    
    Args:
        individual (tuple): The individual to evaluate.
        func (function): The black box function to evaluate.
    
    Returns:
        float: The fitness of the individual.
    """
    
    # Evaluate the function at the individual
    fitness = func(individual)
    
    return fitness

# ```python
# Initialize the optimizer with a budget and dimensionality
optimizer = BlackBoxOptimizer(budget=100, dim=5)

# Optimize a black box function using the optimizer
func = lambda x: x**2
solution, cost = optimizer(__call__, func)

# Print the result
print(f"Optimal solution: {solution}")
print(f"Cost: {cost}")