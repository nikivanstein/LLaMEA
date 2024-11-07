import numpy as np
import random

class TournamentSelectionWithAdaptiveCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.tournament_size = 3
        self.mutation_rate = 0.1
        self.adaptive_crossover = 0.5

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
            best_point = tournament_points[np.argmin(tournament_fitness)]

            # Adaptive crossover
            if np.random.rand() < self.adaptive_crossover:
                crossover_point = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            else:
                crossover_point = np.random.uniform(-0.1, 0.1, size=self.dim)

            child_point = best_point + crossover_point * 0.5

            # Mutation
            if np.random.rand() < self.mutation_rate:
                child_point += np.random.uniform(-0.1, 0.1, size=self.dim)

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
optimizer = TournamentSelectionWithAdaptiveCrossover(budget, dim)
best_point, best_fitness = optimizer(func)
print("Best point:", best_point)
print("Best fitness:", best_fitness)

# Refine the strategy:
# 1. Increase the tournament size to 5.
# 2. Decrease the mutation rate to 0.05.
# 3. Increase the adaptive crossover probability to 0.8.
# 4. Decrease the crossover point's range to (-0.05, 0.05).
# 5. Add a new attribute to store the best fitness of the population.
class TournamentSelectionWithAdaptiveCrossoverRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.tournament_size = 5
        self.mutation_rate = 0.05
        self.adaptive_crossover = 0.8
        self.crossover_point_range = (-0.05, 0.05)

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
            best_point = tournament_points[np.argmin(tournament_fitness)]

            # Adaptive crossover
            if np.random.rand() < self.adaptive_crossover:
                crossover_point = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            else:
                crossover_point = np.random.uniform(self.crossover_point_range[0], self.crossover_point_range[1], size=self.dim)

            child_point = best_point + crossover_point * 0.5

            # Mutation
            if np.random.rand() < self.mutation_rate:
                child_point += np.random.uniform(-0.1, 0.1, size=self.dim)

            # Ensure bounds
            child_point = np.clip(child_point, self.lower_bound, self.upper_bound)

            # Replace worst point
            worst_index = np.argmin(fitness)
            population[worst_index] = child_point
            fitness[worst_index] = func(child_point)

        # Store best fitness
        self.best_fitness = np.min(fitness)

        return population[0], fitness[0], self.best_fitness

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 5
optimizer = TournamentSelectionWithAdaptiveCrossoverRefined(budget, dim)
best_point, fitness, best_fitness = optimizer(func)
print("Best point:", best_point)
print("Best fitness:", best_fitness)
print("Best fitness of population:", best_fitness)