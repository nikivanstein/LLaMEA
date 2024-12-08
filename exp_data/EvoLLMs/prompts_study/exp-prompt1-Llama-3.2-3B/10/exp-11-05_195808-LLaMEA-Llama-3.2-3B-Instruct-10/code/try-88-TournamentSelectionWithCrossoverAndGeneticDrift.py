import numpy as np
import random

class TournamentSelectionWithCrossoverAndGeneticDrift:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

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

            # Genetic drift
            genetic_drift_rate = 0.05
            if np.random.rand() < genetic_drift_rate:
                genetic_drift_point = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
                child_point += genetic_drift_point

            # Ensure bounds
            child_point = np.clip(child_point, self.lower_bound, self.upper_bound)

            # Replace worst point
            worst_index = np.argmin(fitness)
            population[worst_index] = child_point
            fitness[worst_index] = func(child_point)

        # Replace worst 10% of population with new random points
        worst_indices = np.argsort(fitness)[:int(0.1 * self.budget)]
        population[worst_indices] = np.random.uniform(self.lower_bound, self.upper_bound, size=(len(worst_indices), self.dim))
        fitness[worst_indices] = np.array([func(point) for point in population[worst_indices]])

        return population[0], fitness[0]

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 5
optimizer = TournamentSelectionWithCrossoverAndGeneticDrift(budget, dim)
best_point, best_fitness = optimizer(func)
print("Best point:", best_point)
print("Best fitness:", best_fitness)
