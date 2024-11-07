import numpy as np
import random

class TournamentSelectionWithAdaptiveCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.adaptive_crossover_rate = 0.5  # Initialize adaptive crossover rate

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.budget, self.dim))
        fitness = np.array([func(point) for point in population])

        for _ in range(self.budget):
            tournament_size = 3
            tournament_indices = np.random.choice(self.budget, size=tournament_size, replace=False)
            tournament_points = population[tournament_indices]
            tournament_fitness = fitness[tournament_indices]

            best_point = tournament_points[np.argmin(tournament_fitness)]

            if np.random.rand() < self.adaptive_crossover_rate:
                crossover_point = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
                child_point = best_point + crossover_point * 0.5
            else:
                child_point = best_point

            if np.random.rand() < 0.1:  # 10% mutation rate
                mutation_rate = np.random.uniform(-0.1, 0.1, size=self.dim)
                child_point += mutation_rate
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
optimizer = TournamentSelectionWithAdaptiveCrossover(budget, dim)
best_point, best_fitness = optimizer(func)
print("Best point:", best_point)
print("Best fitness:", best_fitness)

# Note: The adaptive crossover rate is adjusted based on the current best point's fitness.
# This adjustment aims to explore the search space more effectively when the current best point is near the optimal solution.