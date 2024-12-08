import random
import numpy as np

class AdaptiveEvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1
        self.evolutionary_history = []
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
                child = random.uniform(self.search_space[0], self.search_space[1])
            new_population.append(child)
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Update the evolutionary history
        self.evolutionary_history.append((new_func_evaluations, new_population))
        
        # Return the best individual
        best_individual = np.argmax(new_func_evaluations)
        self.best_individual = new_population[best_individual]
        
        # Evaluate the best individual with the given budget
        best_func_evaluations = np.array([func(self.best_individual) for _ in range(self.budget)])
        
        # Select the top-performing individuals
        top_individuals = np.argsort(best_func_evaluations)[-self.budget:]
        
        # Create a new population by crossover and mutation
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            new_population.append(child)
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Update the best individual
        best_individual = np.argmax(new_func_evaluations)
        self.best_individual = new_population[best_individual]
        
        return new_population[best_individual]

# Example usage
optimizer = AdaptiveEvolutionaryOptimizer(budget=100, dim=5)
print(optimizer(__call__))