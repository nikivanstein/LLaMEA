import random
import numpy as np

class GeneticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1
        self.evolutionary_strategies = {
            'random': self.random_strategy,
            'elite_crossover': self.elite_crossover_strategy,
            'elite_mutation': self.elite_mutation_strategy
        }

    def random_strategy(self, func):
        # Select the top-performing individuals using random selection
        top_individuals = np.argsort(func(np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)))[:self.population_size]
        return top_individuals

    def elite_crossover_strategy(self, func):
        # Select the top-performing individuals using elitist crossover
        top_individuals = np.argsort(func(np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)))[:self.population_size]
        parent1, parent2 = random.sample(top_individuals, 2)
        child = (parent1 + parent2) / 2
        if random.random() < self.mutation_rate:
            child = random.uniform(self.search_space[0], self.search_space[1])
        return child

    def elite_mutation_strategy(self, func):
        # Select the top-performing individuals using elitist mutation
        top_individuals = np.argsort(func(np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)))[:self.population_size]
        parent1, parent2 = random.sample(top_individuals, 2)
        if random.random() < self.mutation_rate:
            parent1 = random.uniform(self.search_space[0], self.search_space[1])
            parent2 = random.uniform(self.search_space[0], self.search_space[1])
        child = (parent1 + parent2) / 2
        return child

    def __call__(self, func):
        # Evaluate the function with the given budget
        func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
        
        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-self.population_size:]
        
        # Create a new population by crossover and mutation
        new_population = []
        for _ in range(self.population_size):
            if random.random() < self.evolutionary_strategies['random']:
                new_population.append(self.random_strategy(func))
            else:
                new_population.append(self.elite_crossover_strategy(func))
        
        # Replace the old population with the new one
        self.population = new_population
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Return the best individual
        best_individual = np.argmax(new_func_evaluations)
        return new_population[best_individual]

# Description: Black Box Optimization using Genetic Algorithm
# Code: 