import numpy as np
import random

class AdaptiveTournamentSelectionWithCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.tournament_size = 3
        self.mutation_rate = 0.1
        self.adaptation_rate = 0.1

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.budget, self.dim))
        fitness = np.array([func(point) for point in population])

        for _ in range(self.budget):
            tournament_indices = np.random.choice(self.budget, size=self.tournament_size, replace=False)
            tournament_points = population[tournament_indices]
            tournament_fitness = fitness[tournament_indices]

            best_point = tournament_points[np.argmin(tournament_fitness)]

            crossover_point = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            child_point = best_point + crossover_point * 0.5

            if np.random.rand() < self.mutation_rate:
                child_point += np.random.uniform(-0.1, 0.1, size=self.dim)

            child_point = np.clip(child_point, self.lower_bound, self.upper_bound)

            worst_index = np.argmin(fitness)
            population[worst_index] = child_point
            fitness[worst_index] = func(child_point)

            # Adapt tournament size and mutation rate
            if np.mean(fitness) < np.mean(fitness[-int(self.budget * self.adaptation_rate):]):
                self.tournament_size = max(2, int(self.tournament_size * 0.9))
            if np.mean(fitness) > np.mean(fitness[-int(self.budget * self.adaptation_rate):]):
                self.mutation_rate = min(1, self.mutation_rate * 1.1)

        return population[0], fitness[0]

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 5
optimizer = AdaptiveTournamentSelectionWithCrossover(budget, dim)
best_point, best_fitness = optimizer(func)
print("Best point:", best_point)
print("Best fitness:", best_fitness)
