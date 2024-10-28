# Description: Black Box Optimization using Evolutionary Strategies
# Code: 
# ```python
import random
import numpy as np

class BlackBoxOptimizer:
    """
    A metaheuristic algorithm to optimize black box functions.
    
    The algorithm uses a combination of random search, perturbation, and genetic algorithms to find the optimal solution.
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
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        
    def __call__(self, func):
        """
        Optimize the black box function using the optimizer.
        
        Args:
            func (function): The black box function to optimize.
        
        Returns:
            tuple: The optimal solution and its cost.
        """
        
        # Initialize the population with random solutions
        population = self.generate_population(func, self.population_size)
        
        # Run the optimizer for the specified number of iterations
        for _ in range(100):
            # Select the fittest individual
            fittest_individual = self.select_fittest(population)
            
            # Perturb the fittest individual
            perturbed_individual = self.perturb(fittest_individual)
            
            # Evaluate the new individual
            new_cost = func(perturbed_individual)
            
            # Update the fittest individual if the new individual is better
            if new_cost < self.evaluate_fitness(fittest_individual):
                fittest_individual = perturbed_individual
                self.evaluate_fitness(fittest_individual)
        
        # Return the fittest individual and its cost
        return self.evaluate_fitness(fittest_individual), self.evaluate_fitness(fittest_individual)
    
    def generate_population(self, func, population_size):
        """
        Generate a population of random solutions.
        
        Args:
            func (function): The black box function to optimize.
            population_size (int): The number of individuals in the population.
        
        Returns:
            list: A list of random solutions.
        """
        
        # Generate a list of random solutions
        population = []
        for _ in range(population_size):
            individual = (func(np.random.uniform(-5.0, 5.0)), np.random.uniform(-5.0, 5.0))
            population.append(individual)
        
        return population
    
    def select_fittest(self, population):
        """
        Select the fittest individual from the population.
        
        Args:
            population (list): A list of random solutions.
        
        Returns:
            tuple: The fittest individual and its fitness.
        """
        
        # Calculate the fitness of each individual
        fitnesses = [self.evaluate_fitness(individual) for individual in population]
        
        # Select the fittest individual
        fittest_individual = population[np.argmax(fitnesses)]
        
        return fittest_individual, fitnesses[0]
    
    def perturb(self, individual):
        """
        Perturb an individual.
        
        Args:
            individual (tuple): The individual to perturb.
        
        Returns:
            tuple: The perturbed individual.
        """
        
        # Generate a random perturbation
        perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Perturb the individual
        individual = (individual[0] + perturbation[0], individual[1] + perturbation[1])
        
        return individual
    
    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual.
        
        Args:
            individual (tuple): The individual to evaluate.
        
        Returns:
            float: The fitness of the individual.
        """
        
        # Evaluate the fitness of the individual
        fitness = func(individual[0], individual[1])
        
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
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            # Select the fittest individual
            fittest_individual, fitness = self.select_fittest(self.generate_population(func, 100))
            
            # Perturb the fittest individual
            perturbed_individual = self.perturb(fittest_individual)
            
            # Evaluate the new individual
            new_cost = func(perturbed_individual[0], perturbed_individual[1])
            
            # Update the fittest individual if the new individual is better
            if new_cost < fitness:
                fittest_individual = perturbed_individual
                fitness = new_cost
        
        # Return the fittest individual and its cost
        return fittest_individual, fitness