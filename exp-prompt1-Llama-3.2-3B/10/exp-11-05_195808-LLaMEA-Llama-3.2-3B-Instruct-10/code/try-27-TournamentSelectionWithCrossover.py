import numpy as np
import random

class TournamentSelectionWithCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.tournament_size = 3

    def __call__(self, func):
        # Initialize population with random points
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.budget, self.dim))

        # Evaluate population
        fitness = np.array([func(point) for point in population])

        # Selection
        for _ in range(self.budget):
            # Tournament selection
            tournament_indices = np.random.choice(self.budget, size=self.tournament_size, replace=False)
            tournament_points = population[tournament_indices]
            tournament_fitness = fitness[tournament_indices]

            # Get best point
            best_point = np.argmin(tournament_fitness)
            best_point = tournament_points[best_point]

            # Crossover
            crossover_point = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            child_point = best_point + crossover_point * 0.5

            # Mutation with improved probability distribution
            mutation_rate = 0.1
            mutation_distribution = np.random.choice([1, 0.5, 0.1], size=self.dim, replace=True)
            if np.any(mutation_distribution == 1):
                child_point += np.random.uniform(-0.1, 0.1, size=self.dim)
            elif np.any(mutation_distribution == 0.5):
                child_point += np.random.uniform(-0.05, 0.05, size=self.dim)
            elif np.any(mutation_distribution == 0.1):
                child_point += np.random.uniform(-0.05, 0.05, size=self.dim)

            # Ensure bounds
            child_point = np.clip(child_point, self.lower_bound, self.upper_bound)

            # Replace worst point
            worst_index = np.argmin(fitness)
            population[worst_index] = child_point
            fitness[worst_index] = func(child_point)

        return population[0], fitness[0]

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 5
optimizer = TournamentSelectionWithCrossover(budget, dim)
best_point, best_fitness = optimizer(func)
print("Best point:", best_point)
print("Best fitness:", best_fitness)