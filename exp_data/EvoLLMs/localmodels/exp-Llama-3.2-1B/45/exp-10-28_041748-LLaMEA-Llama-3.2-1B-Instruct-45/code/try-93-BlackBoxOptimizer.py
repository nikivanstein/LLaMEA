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
        self.fitness_history = []

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
        
        # Update the fitness history
        self.fitness_history.extend(new_func_evaluations)
        
        # Return the best individual
        best_individual = np.argmax(new_func_evaluations)
        return new_population[best_individual]

    def select_strategies(self, strategies):
        # Select strategies based on fitness history
        selected_strategies = []
        for strategy in strategies:
            # Evaluate the strategy with the given budget
            strategy_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
            
            # Select the top-performing individuals
            top_individuals = np.argsort(strategy_evaluations)[-self.population_size:]
            
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
            
            # Update the fitness history
            self.fitness_history.extend(new_func_evaluations)
            
            # Add the strategy to the selected strategies
            selected_strategies.append(strategy)
        
        # Return the selected strategies
        return selected_strategies

# One-line description with main idea
# Adaptive Black Box Optimization using Evolutionary Strategies
# Selects strategies based on fitness history and evolves the population to improve the overall fitness
# 
# Parameters:
#     budget: int
#     dim: int
# 
# Returns:
#     None