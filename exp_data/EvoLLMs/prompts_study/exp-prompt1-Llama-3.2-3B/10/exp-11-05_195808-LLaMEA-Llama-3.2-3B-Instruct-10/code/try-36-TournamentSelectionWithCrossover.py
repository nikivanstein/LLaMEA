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
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.budget, self.dim))
        fitness = np.array([func(point) for point in population])

        for _ in range(self.budget):
            tournament_indices = np.random.choice(self.budget, size=self.tournament_size, replace=False)
            tournament_points = population[tournament_indices]
            tournament_fitness = fitness[tournament_indices]

            best_point = tournament_points[np.argmin(tournament_fitness)]

            # Adaptive tournament size
            if _ < self.budget * 0.2:
                self.tournament_size = int(self.tournament_size * 1.1)
            elif _ > self.budget * 0.8:
                self.tournament_size = int(self.tournament_size * 0.9)

            crossover_point = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            child_point = best_point + crossover_point * 0.5

            # Improved mutation strategy
            mutation_rate = max(0.01, min(0.1, self.budget / (_ + 1)))
            if np.random.rand() < mutation_rate:
                child_point += np.random.uniform(-0.05, 0.05, size=self.dim)

            child_point = np.clip(child_point, self.lower_bound, self.upper_bound)

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