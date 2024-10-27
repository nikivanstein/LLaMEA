import numpy as np
import random
from copy import deepcopy

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        # Check if the function can be evaluated within the budget
        if self.func_evals >= self.budget:
            raise ValueError("Not enough evaluations left to optimize the function")

        # Evaluate the function within the budget
        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def search(self, func, population_size=100, mutation_rate=0.01):
        # Define the search space
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
        # Initialize the population
        population = [deepcopy(func) for _ in range(population_size)]
        
        # Initialize the best solution
        best_individual = None
        
        # Run the evolutionary algorithm
        for _ in range(1000):
            # Select parents using tournament selection
            parents = []
            for _ in range(population_size):
                tournament_size = random.randint(2, population_size)
                tournament_indices = random.sample(range(population_size), tournament_size)
                tournament_results = [self.__call__(func, individual) for individual in tournament_indices]
                tournament_winner = np.argmax(tournament_results)
                parents.append(deepcopy(population[tournament_winner]))
            
            # Crossover (recombination) the parents
            offspring = []
            for _ in range(population_size):
                parent1, parent2 = random.sample(parents, 2)
                child = (parent1 + parent2) / 2
                if random.random() < mutation_rate:
                    child = (parent1 + parent2 - 0.5) / 1.5
                offspring.append(deepcopy(child))
            
            # Mutate the offspring
            for individual in offspring:
                if random.random() < mutation_rate:
                    individual = self.__call__(func, individual)
            
            # Replace the least fit individuals with the new offspring
            population[:population_size] = offspring
        
        # Return the best solution found
        return self.search(func, population_size=100, mutation_rate=0.01)

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Optimizes a black box function using evolutionary algorithms with tournament selection, crossover, and mutation