# Description: Novel Black Box Optimization Algorithm using Evolutionary Strategies
# Code:
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
        
        # Update the population history
        self.population_history.append((new_population, new_func_evaluations))
        
        # Return the best individual
        best_individual = np.argmax(new_func_evaluations)
        return new_population[best_individual]

    def mutate(self, individual):
        # Randomly select an individual to mutate
        mutated_individual = individual.copy()
        
        # Randomly swap two genes in the individual
        if random.random() < self.mutation_rate:
            index1, index2 = random.sample(range(len(individual)), 2)
            mutated_individual[index1], mutated_individual[index2] = mutated_individual[index2], mutated_individual[index1]
        
        # Randomly scale the individual
        if random.random() < self.mutation_rate:
            mutated_individual = np.clip(mutated_individual, self.search_space[0], self.search_space[1])
        
        return mutated_individual

# Example usage:
optimizer = BlackBoxOptimizer(budget=100, dim=10)
best_individual = optimizer(BlackBoxOptimizer(100, 10))
print(best_individual)