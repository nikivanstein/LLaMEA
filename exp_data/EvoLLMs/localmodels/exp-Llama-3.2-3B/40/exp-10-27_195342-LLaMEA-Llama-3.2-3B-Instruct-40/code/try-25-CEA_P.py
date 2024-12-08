import numpy as np
import random

class CEA_P:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_size = 10
        self.mut_prob = 0.1
        self.crossover_prob = 0.8
        self.adapt_prob = 0.4
        self.mut_prob_elite = 0.1
        self.crossover_prob_elite = 0.5
        self.adapt_prob_elite = 0.6

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
                # Replace the worst individual with the new one with a probability of 0.4
                if random.random() < 0.4:
                    population[np.argmin(scores)] = new_individual

            # Crossover
            if random.random() < self.crossover_prob:
                # Randomly select two individuals
                idx1, idx2 = random.sample(range(self.population_size), 2)
                # Compute the crossover of the two individuals
                child = (population[idx1] + population[idx2]) / 2
                # Replace the worst individual with the child with a probability of 0.5
                if random.random() < 0.5:
                    population[np.argmin(scores)] = child

            # Elitism
            if random.random() < self.elite_size / self.population_size:
                # Replace the worst individual with the best individual with a probability of 0.1
                if random.random() < 0.1:
                    population[np.argmin(scores)] = best

            # Mutation
            if random.random() < self.mut_prob:
                # Randomly select an individual
                idx = random.sample(range(self.population_size), 1)
                # Mutate the individual with a probability of 0.1
                if random.random() < 0.1:
                    population[idx] += np.random.uniform(-1.0, 1.0, self.dim)

        # Return the best individual
        return best, best_func

# Usage
def func(x):
    return sum([i**2 for i in x])

cea_P = CEA_P(budget=100, dim=10)
best, score = cea_P(func)
print(f"Best individual: {best}")
print(f"Best score: {score}")