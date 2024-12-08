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
        self.evolve_strategy = self.evolve_strategy

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

    def evolve_strategy(self, func, budget):
        # Define the evolutionary strategy
        if random.random() < 0.45:
            # Select parents using tournament selection
            tournament_size = int(budget * 0.2)
            tournament_results = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], tournament_size)])
            tournament_indices = np.argsort(tournament_results)[-tournament_size:]
            parent1, parent2 = random.sample(tournament_indices, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            return child
        else:
            # Select parents using genetic algorithm
            population = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
            population = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
            population = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
            while True:
                # Select parents using tournament selection
                tournament_size = int(budget * 0.2)
                tournament_results = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], tournament_size)])
                tournament_indices = np.argsort(tournament_results)[-tournament_size:]
                parent1, parent2 = random.sample(tournament_indices, 2)
                child = (parent1 + parent2) / 2
                if random.random() < self.mutation_rate:
                    child = random.uniform(self.search_space[0], self.search_space[1])
                population = np.array([func(x) for x in population])
                population = np.array([func(x) for x in population])
                if np.max(population) - np.min(population) < 1e-6:
                    break
            return child

# One-line description with the main idea
# Black Box Optimization using Genetic Algorithm with Evolutionary Strategies
# 
# This algorithm uses a genetic algorithm with evolutionary strategies to optimize black box functions.
# The strategy involves selecting parents using tournament selection or genetic algorithm, and then selecting parents using tournament selection again.
# The algorithm evaluates the new population by using a mutation rate and a budget to select the best individual.
# 
# Parameters:
#   budget: The budget for the optimization process
#   dim: The dimensionality of the problem
# 
# Returns:
#   The best individual in the new population