# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from collections import deque
import copy

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1
        self.population_history = deque(maxlen=100)
        self.best_individual = None
        self.best_score = -inf

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
        
        # Update the best individual and score
        if new_func_evaluations.max() > self.best_score:
            self.best_individual = new_population[np.argmax(new_func_evaluations)]
            self.best_score = new_func_evaluations.max()
            self.population_history.append((self.best_individual, self.best_score))
        
        # Return the best individual
        return new_population[0]

    def update(self, func, budget):
        # Check if budget is sufficient
        if budget <= self.budget:
            raise Exception("Not enough budget for optimization")
        
        # Select the top-performing individuals
        top_individuals = np.argsort(func(np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)))[-self.population_size:]
        
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
        
        # Update the best individual and score
        if new_func_evaluations.max() > self.best_score:
            self.best_individual = new_population[np.argmax(new_func_evaluations)]
            self.best_score = new_func_evaluations.max()
            self.population_history.append((self.best_individual, self.best_score))
        
        # Return the best individual
        return new_population[0]

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization
# The algorithm uses a population of individuals with a specified dimension and mutation rate to search for the optimal solution in a black box function.