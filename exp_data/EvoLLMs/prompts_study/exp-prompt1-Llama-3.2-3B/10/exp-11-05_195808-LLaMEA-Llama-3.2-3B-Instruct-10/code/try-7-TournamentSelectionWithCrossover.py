import numpy as np
import random

class TournamentSelectionWithCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.stagnation_detection_threshold = 10  # adjust this value as needed

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.budget, self.dim))
        fitness = np.array([func(point) for point in population])

        for _ in range(self.budget):
            tournament_size = 3
            tournament_indices = np.random.choice(self.budget, size=tournament_size, replace=False)
            tournament_points = population[tournament_indices]
            tournament_fitness = fitness[tournament_indices]

            best_point = tournament_points[np.argmin(tournament_fitness)]

            crossover_point = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            child_point = best_point + crossover_point * 0.5

            mutation_rate = 0.1
            if np.random.rand() < mutation_rate:
                child_point += np.random.uniform(-0.1, 0.1, size=self.dim)

            child_point = np.clip(child_point, self.lower_bound, self.upper_bound)

            worst_index = np.argmin(fitness)
            population[worst_index] = child_point
            fitness[worst_index] = func(child_point)

            # Stagnation detection
            stagnation_count = 0
            for i in range(self.budget):
                if fitness[i] < self.stagnation_detection_threshold:
                    stagnation_count += 1
            if stagnation_count > 2:
                # Replace worst point with a new random point
                population[worst_index] = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
                fitness[worst_index] = func(population[worst_index])

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
