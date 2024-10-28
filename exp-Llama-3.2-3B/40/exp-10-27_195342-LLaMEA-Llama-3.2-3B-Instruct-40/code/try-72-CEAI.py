import numpy as np
import random

class CEAI:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_size = 10
        self.mut_prob = 0.1
        self.crossover_prob = 0.8
        self.adapt_prob = 0.4
        self.improve_prob = 0.2
        self.adaptation_threshold = 0.4
        self.adaptation_threshold_idx = 0
        self.elite_threshold = 0.4
        self.prob_adapt = 0.4

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        best = np.zeros(self.dim)
        best_func = np.inf

        # Differential evolution for optimization
        for _ in range(self.budget):
            # Evaluate population
            scores = [func(x) for x in population]
            for i, score in enumerate(scores):
                if score < best_func:
                    best_func = score
                    best = population[i]

            # Adaptation
            if random.random() < self.adapt_prob:
                self.adaptation_threshold_idx = (self.adaptation_threshold_idx + 1) % self.population_size
                # Randomly select two individuals
                idx1, idx2 = random.sample(range(self.population_size), 2)
                # Compute the average of the two individuals
                new_individual = (population[idx1] + population[idx2]) / 2
                # Replace the worst individual with the new one with probability 0.4
                if random.random() < self.prob_adapt:
                    population[np.argmin(scores)] = new_individual
                    # Check if the new individual is better than the current best
                    if func(new_individual) < best_func:
                        best = new_individual
                        best_func = func(new_individual)

            # Crossover
            if random.random() < self.crossover_prob:
                # Randomly select two individuals
                idx1, idx2 = random.sample(range(self.population_size), 2)
                # Compute the crossover of the two individuals
                child = (population[idx1] + population[idx2]) / 2
                # Replace the worst individual with the child with probability 0.4
                if random.random() < self.prob_adapt:
                    population[np.argmin(scores)] = child
                    # Check if the child is better than the current best
                    if func(child) < best_func:
                        best = child
                        best_func = func(child)

            # Elitism
            if random.random() < self.elite_size / self.population_size:
                # Replace the worst individual with the best individual
                population[np.argmin(scores)] = best

            # Improvement
            if random.random() < self.improve_prob:
                # Select the worst individual
                worst_individual = population[np.argmin(scores)]
                # Generate a new individual by adding a small random perturbation to the worst individual
                new_individual = worst_individual + np.random.uniform(-0.1, 0.1, self.dim)
                # Replace the worst individual with the new one with probability 0.4
                if random.random() < self.prob_adapt:
                    population[np.argmin(scores)] = new_individual
                    # Check if the new individual is better than the current best
                    if func(new_individual) < best_func:
                        best = new_individual
                        best_func = func(new_individual)

        # Return the best individual
        return best, best_func

# Usage
def func(x):
    return sum([i**2 for i in x])

cea = CEAI(budget=100, dim=10)
best, score = cea(func)
print(f"Best individual: {best}")
print(f"Best score: {score}")