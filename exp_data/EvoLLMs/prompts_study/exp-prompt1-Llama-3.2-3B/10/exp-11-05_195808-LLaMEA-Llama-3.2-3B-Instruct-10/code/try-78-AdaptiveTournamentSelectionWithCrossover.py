import numpy as np
import random

class AdaptiveTournamentSelectionWithCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = None

    def __call__(self, func):
        if self.population is None:
            self.population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.budget, self.dim))
            self.fitness = np.array([func(point) for point in self.population])
        else:
            # Replace worst point
            worst_index = np.argmin(self.fitness)
            self.population[worst_index] = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            self.fitness[worst_index] = func(self.population[worst_index])

        # Selection
        for _ in range(self.budget):
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(self.budget, size=tournament_size, replace=False)
            tournament_points = self.population[tournament_indices]
            tournament_fitness = self.fitness[tournament_indices]

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
            worst_index = np.argmin(self.fitness)
            self.population[worst_index] = child_point
            self.fitness[worst_index] = func(child_point)

        return self.population[0], self.fitness[0]

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 5
optimizer = AdaptiveTournamentSelectionWithCrossover(budget, dim)
best_point, best_fitness = optimizer(func)
print("Best point:", best_point)
print("Best fitness:", best_fitness)