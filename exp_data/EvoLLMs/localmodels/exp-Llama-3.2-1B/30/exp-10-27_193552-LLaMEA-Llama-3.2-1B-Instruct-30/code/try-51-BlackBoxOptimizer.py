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
        self.population_size = 100
        self.mutation_rate = 0.01
        self.elite_size = 10
        
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
        
        # Initialize the population
        population = self.generate_population(func, num_iterations)
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            # Select the fittest individual
            fittest_individual = self.select_fittest(population)
            
            # Mutate the fittest individual
            mutated_individual = self.mutate(fittest_individual)
            
            # Evaluate the mutated individual
            new_individual, new_cost = self.evaluate_individual(mutated_individual)
            
            # Update the population
            population = self.update_population(population, new_individual, new_cost)
        
        return fittest_individual, cost = self.evaluate_fitness(population)
        
        # Select the elite individuals
        elite_individuals = self.select_elite(population, self.elite_size)
        
        # Refine the elite individuals
        refined_individuals = self.refine_elite(elite_individuals)
        
        # Update the population
        population = self.update_population(population, refined_individuals, cost)
        
        return refined_individuals, cost
    
    def generate_population(self, func, num_iterations):
        """
        Generate a population of individuals.
        
        Args:
            func (function): The black box function to optimize.
            num_iterations (int): The number of iterations to run.
        
        Returns:
            list: The population of individuals.
        """
        
        # Initialize the population
        population = []
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            # Generate a new individual
            new_individual = self.evaluate_fitness(func)
            
            # Add the new individual to the population
            population.append(new_individual)
        
        return population
    
    def select_fittest(self, population):
        """
        Select the fittest individual.
        
        Args:
            population (list): The population of individuals.
        
        Returns:
            tuple: The fittest individual.
        """
        
        # Calculate the fitness of each individual
        fitness = [self.evaluate_fitness(individual) for individual in population]
        
        # Select the fittest individual
        fittest_individual = population[np.argmax(fitness)]
        
        return fittest_individual
    
    def mutate(self, individual):
        """
        Mutate an individual.
        
        Args:
            individual (tuple): The individual to mutate.
        
        Returns:
            tuple: The mutated individual.
        """
        
        # Generate a random mutation
        mutation = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Mutate the individual
        mutated_individual = (individual[0] + mutation[0], individual[1] + mutation[1])
        
        return mutated_individual
    
    def evaluate_individual(self, individual):
        """
        Evaluate an individual.
        
        Args:
            individual (tuple): The individual to evaluate.
        
        Returns:
            tuple: The fitness of the individual.
        """
        
        # Evaluate the individual using the function
        fitness = self.evaluate_fitness(individual)
        
        return fitness
    
    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual.
        
        Args:
            individual (tuple): The individual to evaluate.
        
        Returns:
            float: The fitness of the individual.
        """
        
        # Evaluate the individual using the function
        fitness = func(individual)
        
        return fitness
    
    def update_population(self, population, new_individual, new_cost):
        """
        Update the population.
        
        Args:
            population (list): The population of individuals.
            new_individual (tuple): The new individual to add.
            new_cost (float): The cost of the new individual.
        
        Returns:
            list: The updated population.
        """
        
        # Add the new individual to the population
        population.append(new_individual)
        
        # Update the cost of the new individual
        population[-1] = (population[-1][0] + new_cost, population[-1][1])
        
        return population
    
    def select_elite(self, population, elite_size):
        """
        Select the elite individuals.
        
        Args:
            population (list): The population of individuals.
            elite_size (int): The size of the elite.
        
        Returns:
            list: The elite individuals.
        """
        
        # Select the elite individuals
        elite_individuals = population[:elite_size]
        
        return elite_individuals
    
    def refine_elite(self, elite_individuals):
        """
        Refine the elite individuals.
        
        Args:
            elite_individuals (list): The elite individuals.
        
        Returns:
            list: The refined elite individuals.
        """
        
        # Initialize the refined elite individuals
        refined_individuals = elite_individuals[:]
        
        # Refine the elite individuals
        for _ in range(100):
            # Select the fittest individual
            fittest_individual = self.select_fittest(refined_individuals)
            
            # Mutate the fittest individual
            mutated_individual = self.mutate(fittest_individual)
            
            # Evaluate the mutated individual
            new_individual, new_cost = self.evaluate_individual(mutated_individual)
            
            # Update the refined elite individuals
            refined_individuals = self.update_population(refined_individuals, new_individual, new_cost)
        
        return refined_individuals
    
    def evaluate_fitness(self, population):
        """
        Evaluate the fitness of the population.
        
        Args:
            population (list): The population of individuals.
        
        Returns:
            tuple: The fitness of the population.
        """
        
        # Evaluate the fitness of each individual
        fitness = [self.evaluate_fitness(individual) for individual in population]
        
        return fitness
    
    def run(self, func, num_iterations):
        """
        Run the optimizer for a specified number of iterations.
        
        Args:
            func (function): The black box function to optimize.
            num_iterations (int): The number of iterations to run.
        
        Returns:
            tuple: The optimal solution and its cost.
        """
        
        # Initialize the population
        population = self.generate_population(func, num_iterations)
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            # Select the fittest individual
            fittest_individual = self.select_fittest(population)
            
            # Mutate the fittest individual
            mutated_individual = self.mutate(fittest_individual)
            
            # Evaluate the mutated individual
            new_individual, new_cost = self.evaluate_individual(mutated_individual)
            
            # Update the population
            population = self.update_population(population, new_individual, new_cost)
        
        # Select the elite individuals
        elite_individuals = self.select_elite(population, self.elite_size)
        
        # Refine the elite individuals
        refined_individuals = self.refine_elite(elite_individuals)
        
        # Update the population
        population = self.update_population(population, refined_individuals, self.evaluate_fitness(population))
        
        return refined_individuals, self.evaluate_fitness(population)

# Description: Black Box Optimization Algorithm using Evolutionary Strategies
# Code: 