import numpy as np
import random

class MultiPhaseEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.crossover_prob = 0.8
        self.mutation_prob = 0.1
        self.adaptation_rate = 0.2

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        # Multi-phase evolution
        for phase in range(5):
            # Selection
            fitness = np.array([func(x) for x in population])
            selection = np.argsort(fitness)
            population = population[selection[:int(self.population_size * 0.8)]]

            # Crossover
            offspring = []
            while len(offspring) < self.population_size - len(population):
                parent1, parent2 = random.sample(population, 2)
                child = parent1 + parent2 * (np.random.uniform(-1.0, 1.0, self.dim) * (parent2 - parent1))
                if np.random.rand() < self.crossover_prob:
                    child = np.random.uniform(-5.0, 5.0, self.dim)
                offspring.append(child)

            # Mutation
            for i in range(len(offspring)):
                if np.random.rand() < self.mutation_prob:
                    offspring[i] += np.random.uniform(-1.0, 1.0, self.dim)

            # Adaptation
            if phase < 4:
                # Replace worst 20% of the population
                population = np.array([x for x in population if x < np.min(population, axis=0) + self.adaptation_rate * (np.max(population, axis=0) - np.min(population, axis=0))])
                population = np.array([x for x in population if np.random.rand() < 0.8])

            # Update population
            population = np.array([x for x in population if func(x) < func(offspring[i])])
            population = np.vstack((population, offspring))

        # Evaluate the best individual
        best_individual = np.argmin([func(x) for x in population])
        return population[best_individual]

# Example usage:
budget = 100
dim = 10
func = lambda x: np.sum(x**2)
optimizer = MultiPhaseEvolutionaryAlgorithm(budget, dim)
best_individual = optimizer(func)
print("Best individual:", best_individual)
print("Best fitness:", func(best_individual))