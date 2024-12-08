import numpy as np
import random

class TournamentSelectionWithCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.diversity_threshold = 0.5

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

            # Enhanced mutation
            mutation_rate = 0.1
            if np.random.rand() < mutation_rate:
                # Add a small random value to the child point to enhance diversity
                child_point += np.random.uniform(-0.05, 0.05, size=self.dim)

            # Ensure bounds
            child_point = np.clip(child_point, self.lower_bound, self.upper_bound)

            # Calculate diversity of the new point
            diversity = np.mean(np.abs(child_point - population))

            # If diversity is above threshold, replace worst point
            if diversity > self.diversity_threshold:
                worst_index = np.argmin(fitness)
                population[worst_index] = child_point
                fitness[worst_index] = func(child_point)

        # Use a weighted average of the best points to improve convergence
        best_points = [point for _, point in sorted(zip(fitness, population), reverse=True)]
        best_points = np.array(best_points)
        best_points = (best_points - self.lower_bound) / (self.upper_bound - self.lower_bound)
        best_points = np.mean(best_points, axis=0) * (self.upper_bound - self.lower_bound) + self.lower_bound

        return best_points, fitness[0]

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 5
optimizer = TournamentSelectionWithCrossover(budget, dim)
best_point, best_fitness = optimizer(func)
print("Best point:", best_point)
print("Best fitness:", best_fitness)