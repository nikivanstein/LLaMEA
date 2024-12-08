import numpy as np
import random

class ProbabilisticEvolutionStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_prob = 0.3
        self.population_size = 100
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            scores = np.array([func(x) for x in self.population])
            best_idx = np.argmin(scores)
            best_individual = self.population[best_idx]
            new_population = np.copy(self.population)
            new_population[best_idx] = self.mutate(best_individual)
            scores = np.array([func(x) for x in new_population])
            if np.random.rand() < self.mutation_prob:
                new_population = self.mutate(new_population)
            self.population = new_population
        return self.population[np.argmin([func(x) for x in self.population])]

    def mutate(self, individual):
        mutated_individual = individual.copy()
        for i in range(self.dim):
            if np.random.rand() < self.mutation_prob:
                mutated_individual[i] += np.random.uniform(-1.0, 1.0)
                mutated_individual[i] = np.clip(mutated_individual[i], self.lower_bound, self.upper_bound)
        return mutated_individual

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
es = ProbabilisticEvolutionStrategy(budget, dim)
optimal_x = es(func)
print(optimal_x)