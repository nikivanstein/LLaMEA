import random
import numpy as np
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
        return new_population[best_individual]

    def optimize(self, func, max_iter=1000, tolerance=1e-6):
        # Initialize the population
        population = [func(np.random.uniform(self.search_space[0], self.search_space[1])) for _ in range(self.population_size)]
        
        # Run the optimization
        for _ in range(max_iter):
            # Evaluate the fitness of each individual
            fitness = [self(func(x)) for x in np.random.uniform(self.search_space[0], self.search_space[1])]
            
            # Select the fittest individuals
            top_individuals = np.argsort(fitness)[-self.population_size:]
            
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
            new_fitness = [self(func(x)) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))]
            
            # Check for convergence
            if np.allclose(fitness, new_fitness, atol=tolerance):
                break
        
        # Return the best individual
        return population[np.argmax(fitness)]

# Description: Evolutionary Optimization of Black Box Functions
# Code: 