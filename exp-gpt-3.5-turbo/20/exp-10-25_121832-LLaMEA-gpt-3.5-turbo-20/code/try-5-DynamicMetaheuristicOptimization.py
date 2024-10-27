import numpy as np

class DynamicMetaheuristicOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9
        self.archive_size = 10

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

        def mutate(target, population):
            indices = np.random.choice(len(population), 3, replace=False)
            donor = population[indices[0]] + self.mutation_factor * (population[indices[1]] - population[indices[2]])
            return donor

        def crossover(target, donor):
            child = np.where(np.random.rand(self.dim) < self.crossover_prob, donor, target)
            return child

        population = initialize_population()
        archive = np.zeros((self.archive_size, self.dim))
        evaluations = 0

        while evaluations < self.budget:
            for idx, target in enumerate(population):
                donor = mutate(target, population)
                child = crossover(target, donor)

                if func(child) < func(target):
                    population[idx] = child

                archive = np.vstack((archive, target))
                archive = archive[np.argsort([func(individual) for individual in archive])[:self.archive_size]]

                evaluations += 1
                if evaluations >= self.budget:
                    break

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution