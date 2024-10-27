import numpy as np
import random

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]
        self.best_individual = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_score = -np.inf
        self.temperature = 1000.0
        self.cooling_rate = 0.99

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the population
            scores = [func(individual) for individual in self.population]
            # Select the best individual
            best_individual_index = np.argmax(scores)
            self.best_individual = self.population[best_individual_index]
            self.best_score = scores[best_individual_index]

            # Update the population using simulated annealing and particle swarm optimization
            new_population = []
            for _ in range(self.population_size):
                # Generate a new individual using simulated annealing
                new_individual = self.population[best_individual_index]
                while True:
                    new_individual = new_individual + np.random.normal(0, 1)
                    new_individual = np.clip(new_individual, -5.0, 5.0)
                    new_individual_score = func(new_individual)
                    if new_individual_score > self.best_score:
                        break
                # Update the new individual using particle swarm optimization
                new_individual = new_individual + np.random.uniform(-1, 1, self.dim)
                new_individual = np.clip(new_individual, -5.0, 5.0)
                new_individual_score = func(new_individual)
                if new_individual_score > self.best_score:
                    new_individual = new_individual + np.random.normal(0, 1)
                    new_individual = np.clip(new_individual, -5.0, 5.0)
                new_individual_score = func(new_individual)
                if new_individual_score > self.best_score:
                    break
                new_individual = new_individual + np.random.normal(0, 1)
                new_individual = np.clip(new_individual, -5.0, 5.0)
                new_individual_score = func(new_individual)
                if new_individual_score > self.best_score:
                    break
                new_population.append(new_individual)
            self.population = new_population
            # Cool down the temperature
            self.temperature *= self.cooling_rate

# Example usage
def func(x):
    return np.sum(x**2)

hybrid = HybridMetaheuristic(budget=100, dim=5)
hybrid()