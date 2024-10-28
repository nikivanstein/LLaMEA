import random
import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1
        self.best_individual = None
        self.best_score = float('-inf')

    def __call__(self, func):
        # Evaluate the function with the given budget
        func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
        
        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-self.population_size:]
        
        # Create a new population by evolutionary crossover and mutation
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            # Apply evolutionary crossover with adaptive strategy
            if random.random() < 0.5:
                child = self.crossover_adaptive(child)
            else:
                child = self.crossover_nonadaptive(child)
            new_population.append(child)
        
        # Replace the old population with the new one
        self.population = new_population
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Update the best individual and score
        best_individual = np.argmax(new_func_evaluations)
        self.best_individual = best_individual
        self.best_score = new_func_evaluations[best_individual]
        
        # Return the best individual
        return new_population[best_individual]

    def crossover_adaptive(self, parent1, parent2):
        # Apply adaptive crossover with probability 0.45
        if random.random() < 0.45:
            return self.crossover_nonadaptive(parent1)
        else:
            return self.crossover_nonadaptive(parent2)

    def crossover_nonadaptive(self, parent1, parent2):
        # Apply non-adaptive crossover
        return (parent1 + parent2) / 2

# Description: Adaptive Black Box Optimization using Genetic Algorithm with Evolutionary Crossover and Mutation
# Code: 