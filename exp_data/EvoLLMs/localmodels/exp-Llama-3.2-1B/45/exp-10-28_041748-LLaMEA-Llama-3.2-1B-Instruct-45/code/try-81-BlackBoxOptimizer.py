import random
import numpy as np

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
        return new_population[best_individual]

    def optimize(self, func, budget, dim):
        # Create an empty population
        population = []
        
        # Repeat the optimization process until the budget is exhausted
        while budget > 0:
            # Evaluate the function with the current population
            func_evaluations = np.array([func(x) for x in population])
            
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
            population = new_population
            
            # Evaluate the new population
            new_func_evaluations = np.array([func(x) for x in population])
            
            # Update the budget
            budget -= np.sum(new_func_evaluations)
        
        # Return the best individual
        return population[np.argmax(new_func_evaluations)]

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# Optimizes black box functions using a novel combination of crossover and mutation strategies