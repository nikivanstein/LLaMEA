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
        
        # Update the best individual and its fitness
        best_individual = np.argmax(new_func_evaluations)
        best_fitness = new_func_evaluations[best_individual]
        self.population_history.append((best_individual, best_fitness))
        
        # Return the best individual
        return new_population[best_individual]

    def get_best_solution(self):
        if not self.population_history:
            return None
        return self.population_history[-1][0]

    def select_best_solution(self):
        return self.get_best_solution()

    def update_best_solution(self):
        return self.select_best_solution()

    def __str__(self):
        return "BlackBoxOptimizer: Evolutionary Algorithm for Black Box Optimization"

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization
# 
# This algorithm uses a population of candidate solutions to search for the optimal solution.
# It uses a combination of crossover and mutation to generate new solutions and evaluates their fitness.
# The best solution is selected and its fitness is updated.
# 
# Code: 