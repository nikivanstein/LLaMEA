import random
import numpy as np
import copy
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1

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
        return best_individual

    def optimize(self, func, budget, dim):
        # Initialize the population
        population = [random.uniform(self.search_space[0], self.search_space[1]) for _ in range(self.population_size)]
        
        # Initialize the logger
        aoc_logger = None
        if budget > 0:
            aoc_logger = minimize(lambda x: func(x), population, method="SLSQP", bounds=[(self.search_space[0], self.search_space[1])], constraints={"type": "eq", "fun": lambda x: 0}, options={"maxiter": 1000})
        
        # Run the optimization algorithm
        for _ in range(100):
            # Select the top-performing individuals
            top_individuals = np.argsort(func(population))[-self.population_size:]
            
            # Create a new population by crossover and mutation
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(top_individuals, 2)
                child = (parent1 + parent2) / 2
                if random.random() < self.mutation_rate:
                    child = random.uniform(self.search_space[0], self.search_space[1])
                new_population.append(child)
            
            # Replace the old population with the new one
            population = new_population
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(population))])
        
        # Return the best individual
        best_individual = np.argmax(new_func_evaluations)
        return best_individual

# One-line description: 
# A novel black box optimization algorithm using evolutionary strategies to find the optimal solution in a high-dimensional search space.
# 
# Code: 