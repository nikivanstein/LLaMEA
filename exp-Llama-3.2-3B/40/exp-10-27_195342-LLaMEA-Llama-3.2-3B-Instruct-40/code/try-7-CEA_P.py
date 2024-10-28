import numpy as np
import random
from scipy.optimize import differential_evolution

class CEA_P:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_size = 10
        self.mut_prob = 0.1
        self.crossover_prob = 0.8
        self.adapt_prob = 0.4
        self.prob_adj = 0.4

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
                # Randomly select two individuals
                idx1, idx2 = random.sample(range(self.population_size), 2)
                # Compute the average of the two individuals
                new_individual = (population[idx1] + population[idx2]) / 2
                # Replace the worst individual with the new one with probability (1 - self.prob_adj)
                if random.random() > self.prob_adj:
                    population[np.argmin(scores)] = new_individual

            # Crossover
            if random.random() < self.crossover_prob:
                # Randomly select two individuals
                idx1, idx2 = random.sample(range(self.population_size), 2)
                # Compute the crossover of the two individuals
                child = (population[idx1] + population[idx2]) / 2
                # Replace the worst individual with the child with probability (1 - self.prob_adj)
                if random.random() > self.prob_adj:
                    population[np.argmin(scores)] = child

            # Elitism
            if random.random() < self.elite_size / self.population_size:
                # Replace the worst individual with the best individual
                population[np.argmin(scores)] = best

        # Adjust probability based on the number of successful adaptations and crossovers
        self.prob_adj = min(self.prob_adj + self.adapt_prob * (self.adapt_prob / self.budget), 1)
        self.prob_adj = max(self.prob_adj - self.adapt_prob * (self.adapt_prob / self.budget), 0)

        # Return the best individual
        return best, best_func

# Usage
def func(x):
    return sum([i**2 for i in x])

cea_p = CEA_P(budget=100, dim=10)
best, score = cea_p(func)
print(f"Best individual: {best}")
print(f"Best score: {score}")