import random
import numpy as np

class AdaptiveMutationExp:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1
        self.best_individual = None

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
                if random.random() < self.mutation_rate * 0.7:  # 70% of mutation rate
                    child = random.uniform(self.search_space[0], self.search_space[1])
                else:
                    child = random.uniform(self.search_space[0], self.search_space[1])
            new_population.append(child)
        
        # Replace the old population with the new one
        self.population = new_population
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Update the best individual
        self.best_individual = np.argmax(new_func_evaluations)
        
        # Evaluate the best individual
        best_individual = np.argmax(func_evaluations)
        
        # Return the best individual
        return new_population[best_individual]

# Description: Evolutionary Algorithm with Adaptive Mutation Rate
# Code: 