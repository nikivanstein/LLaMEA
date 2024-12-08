import numpy as np

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.mutation_rate = 0.1
        self.crossover_prob = 0.8

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

        def mutate(child):
            mutation_mask = np.random.rand(self.dim) < self.mutation_rate
            child[mutation_mask] += np.random.uniform(-0.5, 0.5, np.sum(mutation_mask))
            return child

        def crossover(parent1, parent2):
            mask = np.random.rand(self.dim) < self.crossover_prob
            child = parent1.copy()
            child[mask] = parent2[mask]
            return child

        population = initialize_population()
        evaluations = 0

        while evaluations < self.budget:
            sorted_indices = np.argsort([func(individual) for individual in population])
            parents = population[sorted_indices[:2]]

            child = crossover(parents[0], parents[1])
            child = mutate(child)

            if func(child) < func(parents[sorted_indices[0]]):
                population[sorted_indices[-1]] = child

            evaluations += 1

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution