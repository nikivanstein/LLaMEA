import numpy as np
import random

class TournamentSelectionWithCrossoverAndElitism:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.elitism_rate = 0.2  # New parameter for elitism rate

    def __call__(self, func):
        # Initialize population with random points
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.budget, self.dim))

        # Evaluate population
        fitness = np.array([func(point) for point in population])

        # Selection
        for _ in range(self.budget):
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(self.budget, size=tournament_size, replace=False)
            tournament_points = population[tournament_indices]
            tournament_fitness = fitness[tournament_indices]

            # Get best point
            best_point = tournament_points[np.argmin(tournament_fitness)]

            # Crossover
            crossover_point = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            child_point = best_point + crossover_point * 0.5

            # Mutation
            mutation_rate = 0.1
            if np.random.rand() < mutation_rate:
                child_point += np.random.uniform(-0.1, 0.1, size=self.dim)

            # Ensure bounds
            child_point = np.clip(child_point, self.lower_bound, self.upper_bound)

            # Replace worst point
            worst_index = np.argmin(fitness)
            population[worst_index] = child_point
            fitness[worst_index] = func(child_point)

            # Add elitism
            if np.random.rand() < self.elitism_rate:
                population[0] = child_point
                fitness[0] = func(child_point)

        # Replace worst point with best point
        worst_index = np.argmin(fitness)
        population[worst_index] = population[0]
        fitness[worst_index] = fitness[0]

        return population[0], fitness[0]

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 5
optimizer = TournamentSelectionWithCrossoverAndElitism(budget, dim)
best_point, best_fitness = optimizer(func)
print("Best point:", best_point)
print("Best fitness:", best_fitness)

# Note: The code only changed by 10% as requested.
# The new code added an elitism rate parameter and added elitism to the selection process.