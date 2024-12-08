import numpy as np
import random

class CEAI:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_size = 10
        self.mut_prob = 0.1
        self.crossover_prob = 0.7
        self.adapt_prob = 0.3
        self.improve_prob = 0.2
        self.adaptation_threshold = 0.4
        self.adaptation_threshold_idx = 0
        self.elite_threshold = 0.4
        self.prob_adapt_individual = 0.3  # Changed probability to 0.3
        self.prob_mut_individual = 0.3  # Changed probability to 0.3
        self.prob_improve_individual = 0.3  # Changed probability to 0.3
        self.prob_adapt = 0.3  # Changed probability to 0.3

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        best = np.zeros(self.dim)
        best_func = np.inf

        for _ in range(self.budget):
            scores = [func(x) for x in population]
            for i, score in enumerate(scores):
                if score < best_func:
                    best_func = score
                    best = population[i]

            if random.random() < self.adapt_prob:
                self.adaptation_threshold_idx = (self.adaptation_threshold_idx + 1) % self.population_size
                idx1, idx2 = random.sample(range(self.population_size), 2)
                new_individual = (population[idx1] + population[idx2]) / 2
                population[np.argmin(scores)] = new_individual
                if func(new_individual) < best_func:
                    best = new_individual
                    best_func = func(new_individual)

            if random.random() < self.crossover_prob:
                idx1, idx2 = random.sample(range(self.population_size), 2)
                child = (population[idx1] + population[idx2]) / 2
                population[np.argmin(scores)] = child
                if func(child) < best_func:
                    best = child
                    best_func = func(child)

            if random.random() < self.elite_size / self.population_size:
                population[np.argmin(scores)] = best

            if random.random() < self.improve_prob:
                worst_individual = population[np.argmin(scores)]
                new_individual = worst_individual + np.random.uniform(-0.1, 0.1, self.dim)
                population[np.argmin(scores)] = new_individual
                if func(new_individual) < best_func:
                    best = new_individual
                    best_func = func(new_individual)

            if random.random() < self.prob_mut_individual:
                idx = random.randint(0, self.population_size - 1)
                new_individual = population[idx] + np.random.uniform(-0.1, 0.1, self.dim)
                population[idx] = new_individual
                if func(new_individual) < best_func:
                    best = new_individual
                    best_func = func(new_individual)

            if random.random() < self.prob_improve_individual:
                worst_individual = population[np.argmin(scores)]
                new_individual = worst_individual + np.random.uniform(-0.1, 0.1, self.dim)
                population[np.argmin(scores)] = new_individual
                if func(new_individual) < best_func:
                    best = new_individual
                    best_func = func(new_individual)

            if random.random() < self.prob_adapt:
                idx1, idx2 = random.sample(range(self.population_size), 2)
                new_individual = (population[idx1] + population[idx2]) / 2
                population[np.argmin(scores)] = new_individual
                if func(new_individual) < best_func:
                    best = new_individual
                    best_func = func(new_individual)

        return best, best_func

# Usage
def func(x):
    return sum([i**2 for i in x])

cea = CEAI(budget=100, dim=10)
best, score = cea(func)
print(f"Best individual: {best}")
print(f"Best score: {score}")