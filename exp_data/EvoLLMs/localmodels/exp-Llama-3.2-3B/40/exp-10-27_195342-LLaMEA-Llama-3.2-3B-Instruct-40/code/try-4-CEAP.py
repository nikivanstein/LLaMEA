import numpy as np
import random

class CEAP:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_size = 10
        self.mut_prob = 0.1
        self.crossover_prob = 0.8
        self.adapt_prob = 0.4
        self.prob_adjustment = 0.1

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
                # Replace the worst individual with the new one
                population[np.argmin(scores)] = new_individual

            # Crossover
            if random.random() < self.crossover_prob:
                # Randomly select two individuals
                idx1, idx2 = random.sample(range(self.population_size), 2)
                # Compute the crossover of the two individuals
                child = (population[idx1] + population[idx2]) / 2
                # Replace the worst individual with the child
                population[np.argmin(scores)] = child

            # Elitism
            if random.random() < self.elite_size / self.population_size:
                # Replace the worst individual with the best individual
                population[np.argmin(scores)] = best

            # Probability adjustment
            if random.random() < self.prob_adjustment:
                # Randomly adjust the probability of adaptation, crossover, and elitism
                self.adapt_prob *= (1 + random.random() * 0.2 - 0.1)
                self.crossover_prob *= (1 + random.random() * 0.2 - 0.1)
                self.elite_size *= (1 + random.random() * 0.2 - 0.1)

        # Return the best individual
        return best, best_func

# Usage
def func(x):
    return sum([i**2 for i in x])

cea = CEAP(budget=100, dim=10)
best, score = cea(func)
print(f"Best individual: {best}")
print(f"Best score: {score}")