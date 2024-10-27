import numpy as np

class QuantumEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_prob = 0.2

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        def quantum_bit_flip(individual):
            mutated_individual = individual.copy()
            for i in range(self.dim):
                if np.random.rand() < self.mutation_prob:
                    mutated_individual[i] = 5.0 - mutated_individual[i]
            return mutated_individual

        population = initialize_population()
        evaluations = 0

        while evaluations < self.budget:
            for idx, target in enumerate(population):
                mutated_individual = quantum_bit_flip(target)

                if func(mutated_individual) < func(target):
                    population[idx] = mutated_individual

                evaluations += 1
                if evaluations >= self.budget:
                    break

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution