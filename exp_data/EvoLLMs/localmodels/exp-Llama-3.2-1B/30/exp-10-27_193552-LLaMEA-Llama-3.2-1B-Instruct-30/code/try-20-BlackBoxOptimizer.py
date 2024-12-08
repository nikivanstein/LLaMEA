import random
import numpy as np
import operator
import copy

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
        self.population = self.initialize_population()
        self.population = self.population[:self.budget]  # Limit the population to the budget
        
    def initialize_population(self):
        """
        Initialize the population with random solutions.
        
        Returns:
            list: The initialized population.
        """
        
        population = []
        for _ in range(self.budget):
            individual = tuple(random.uniform(-5.0, 5.0) for _ in range(self.dim))
            population.append(individual)
        
        return population
    
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
        
        # Perform genetic programming
        for _ in range(self.budget):
            # Select parents using tournament selection
            parents = self.select_parents()
            
            # Crossover (reproduce) the parents
            offspring = self.crossover(parents)
            
            # Mutate the offspring
            offspring = self.mutate(offspring)
            
            # Evaluate the new solution
            new_cost = func(offspring[0])
            
            # Update the solution and cost if the new solution is better
            if new_cost < cost:
                solution = offspring[0]
                cost = new_cost
        
        return solution, cost
    
    def select_parents(self):
        """
        Select parents using tournament selection.
        
        Returns:
            list: The selected parents.
        """
        
        # Select two parents
        parents = []
        for _ in range(2):
            parent1 = copy.deepcopy(self.population)
            parent2 = copy.deepcopy(self.population)
            
            # Select a random parent
            parent = random.choice([parent1, parent2])
            
            # Evaluate the fitness of the parent
            fitness = self.evaluate_fitness(parent)
            
            # Add the parent with the higher fitness
            parents.append((parent, fitness))
        
        # Sort the parents by fitness
        parents.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top two parents
        return [parent[0] for parent in parents[:2]]
    
    def crossover(self, parents):
        """
        Crossover (reproduce) the parents.
        
        Args:
            parents (list): The parents.
        
        Returns:
            list: The offspring.
        """
        
        # Select two parents
        parent1, parent2 = parents
        
        # Create a new offspring
        offspring = []
        
        # Generate a random crossover point
        crossover_point = random.randint(1, self.dim)
        
        # Create the offspring
        for i in range(self.dim):
            if i < crossover_point:
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])
        
        return offspring
    
    def mutate(self, offspring):
        """
        Mutate the offspring.
        
        Args:
            offspring (list): The offspring.
        
        Returns:
            list: The mutated offspring.
        """
        
        # Mutate each individual
        mutated_offspring = []
        
        for individual in offspring:
            # Generate a random mutation
            mutation = (random.uniform(-1, 1), random.uniform(-1, 1))
            
            # Mutate the individual
            mutated_individual = tuple(x + mutation[0] for x in individual)
            
            # Add the mutated individual to the list
            mutated_offspring.append(mutated_individual)
        
        return mutated_offspring
    
    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual.
        
        Args:
            individual (tuple): The individual.
        
        Returns:
            float: The fitness.
        """
        
        # Evaluate the fitness of the individual
        fitness = func(individual)
        
        # Return the fitness
        return fitness
    
# Description: Improved Randomized Black Box Optimization Algorithm using Genetic Programming
# Code: 