import numpy as np
import random

class CEAProb:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_size = 10
        self.mut_prob = 0.1
        self.crossover_prob = 0.8
        self.adapt_prob = 0.4
        self.mut_prob_elite = 0.2
        self.mut_prob_non_elite = 0.05

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
                if random.random() < self.mut_prob_elite:
                    population[np.argmin(scores)] = new_individual
                else:
                    population[np.argmin(scores)] = population[idx1]

            # Crossover
            if random.random() < self.crossover_prob:
                # Randomly select two individuals
                idx1, idx2 = random.sample(range(self.population_size), 2)
                # Compute the crossover of the two individuals
                child = (population[idx1] + population[idx2]) / 2
                # Replace the worst individual with the child
                if random.random() < self.mut_prob_non_elite:
                    population[np.argmin(scores)] = child
                else:
                    population[np.argmin(scores)] = population[idx2]

            # Elitism
            if random.random() < self.elite_size / self.population_size:
                # Replace the worst individual with the best individual
                population[np.argmin(scores)] = best

        # Return the best individual
        return best, best_func

# Usage
def func(x):
    return sum([i**2 for i in x])

cea_prob = CEAProb(budget=100, dim=10)
best, score = cea_prob(func)
print(f"Best individual: {best}")
print(f"Best score: {score}")