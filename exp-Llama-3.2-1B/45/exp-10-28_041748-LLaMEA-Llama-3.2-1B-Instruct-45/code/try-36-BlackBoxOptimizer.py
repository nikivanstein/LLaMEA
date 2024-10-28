# Description: Novel metaheuristic algorithm for solving black box optimization problems
# Code: 
# ```python
import random
import numpy as np
import math
import copy

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1
        self.population_history = []

    def __call__(self, func):
        # Evaluate the function with the given budget
        func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
        
        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-self.population_size:]
        
        # Create a new population by crossover and mutation
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            new_population.append(child)
        
        # Replace the old population with the new one
        self.population = new_population
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Return the best individual
        best_individual = np.argmax(new_func_evaluations)
        self.population_history.append((best_individual, new_func_evaluations[best_individual]))
        
        # Return the best individual
        return new_population[best_individual]

    def select_parents(self, num_parents):
        # Select parents based on probability 0.45
        parents = []
        for _ in range(num_parents):
            if random.random() < 0.45:
                parents.append(self.population)
            else:
                parents.append(np.random.choice(self.population, size=self.population_size, replace=False))
        
        # Create new parents by crossover and mutation
        new_parents = []
        for _ in range(num_parents):
            parent1, parent2 = random.sample(parents, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            new_parents.append(child)
        
        return new_parents

    def evolve_population(self, num_generations):
        # Evolve the population for the specified number of generations
        for _ in range(num_generations):
            # Select parents
            parents = self.select_parents(self.population_size)
            
            # Create new parents by crossover and mutation
            new_parents = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(parents, 2)
                child = (parent1 + parent2) / 2
                if random.random() < self.mutation_rate:
                    child = random.uniform(self.search_space[0], self.search_space[1])
                new_parents.append(child)
            
            # Replace the old population with the new one
            self.population = new_parents
            
            # Evaluate the new population
            new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_parents))])
            
            # Return the best individual
            best_individual = np.argmax(new_func_evaluations)
            self.population_history.append((best_individual, new_func_evaluations[best_individual]))
        
        return self.population_history

# One-line description with the main idea
# Novel metaheuristic algorithm for solving black box optimization problems
# The algorithm uses a combination of crossover and mutation to evolve a population of individuals, each of which is evaluated multiple times to select the fittest individuals
# The algorithm is designed to handle a wide range of tasks and can be easily adapted to different search spaces and objective functions
# The probability of selecting parents is set to 0.45, and the algorithm evolves the population for a specified number of generations