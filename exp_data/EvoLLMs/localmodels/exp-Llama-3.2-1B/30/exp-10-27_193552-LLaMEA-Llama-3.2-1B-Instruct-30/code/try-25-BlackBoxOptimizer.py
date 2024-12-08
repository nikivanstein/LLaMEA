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
            # Select the fittest solution
            fittest_solution = population[np.argmax([func(s) for s in population])]
            
            # Mutate the fittest solution
            mutated_solution = self.mutate(fittest_solution)
            
            # Evaluate the mutated solution
            new_cost = func(mutated_solution)
            
            # Update the population with the mutated solution
            population = self.update_population(population, new_cost, mutated_solution)
        
        return population[0], self.evaluate_fitness(population[0])
    
    def initialize_population(self, num_iterations):
        """
        Initialize the population with random solutions.
        
        Args:
            num_iterations (int): The number of iterations to run.
        
        Returns:
            list: The population of solutions.
        """
        
        # Initialize the population with random solutions
        population = []
        for _ in range(num_iterations):
            solution = tuple(np.random.uniform(self.search_space[i][0], self.search_space[i][1]) for i in range(self.dim))
            population.append(solution)
        
        return population
    
    def mutate(self, solution):
        """
        Mutate a solution.
        
        Args:
            solution (tuple): The solution to mutate.
        
        Returns:
            tuple: The mutated solution.
        """
        
        # Generate a random mutation in the search space
        mutation = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Update the solution with the mutation
        solution = (solution[0] + mutation[0], solution[1] + mutation[1])
        
        return solution
    
    def update_population(self, population, new_cost, mutated_solution):
        """
        Update the population with a new solution.
        
        Args:
            population (list): The population of solutions.
            new_cost (float): The cost of the new solution.
            mutated_solution (tuple): The mutated solution.
        
        Returns:
            list: The updated population.
        """
        
        # Evaluate the new solution
        new_fitness = new_cost
        
        # Update the population with the new solution
        population = self.population_update(population, new_fitness, mutated_solution)
        
        return population
    
    def population_update(self, population, new_fitness, mutated_solution):
        """
        Update the population with a new solution.
        
        Args:
            population (list): The population of solutions.
            new_fitness (float): The new fitness of the new solution.
            mutated_solution (tuple): The mutated solution.
        
        Returns:
            list: The updated population.
        """
        
        # Calculate the probability of mutation
        mutation_probability = 0.3
        
        # Select the fittest solution
        fittest_solution = population[np.argmax([func(s) for s in population])]
        
        # Mutate the fittest solution
        mutated_solution = self.mutate(fittest_solution)
        
        # Update the population with the mutated solution
        population = self.update_population(population, new_fitness, mutated_solution)
        
        # Return the updated population
        return population

# Description: A novel metaheuristic algorithm for black box optimization problems, 
#               which combines perturbation and mutation to refine the solution strategy.

# Code: 