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
        return best_individual

def mutation(individual, mutation_rate):
    if random.random() < mutation_rate:
        return random.uniform(self.search_space[0], self.search_space[1])
    return individual

def crossover(parent1, parent2):
    return (parent1 + parent2) / 2

def bbob(func, budget, dim):
    optimizer = BlackBoxOptimizer(budget, dim)
    while True:
        best_individual = optimizer.__call__(func)
        best_fitness = np.max([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], 100)])
        
        # Refine the strategy
        mutation_rate = 0.9
        if random.random() < 0.45:
            mutation_rate = 0.95
        new_individual = mutation(best_individual, mutation_rate)
        new_fitness = np.max([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], 100)])
        
        # Replace the old population with the new one
        optimizer.population = [new_individual] * optimizer.population_size
        optimizer.population = optimizer.population[:100]
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(optimizer.population))])
        
        # Check if the new population is better
        if np.max([np.max([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], 100)])] > best_fitness):
            return best_individual
        else:
            return optimizer.__call__(func)

# Description: Evolutionary Algorithm for BBOB Optimization
# Code: 