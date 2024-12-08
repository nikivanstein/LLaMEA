import numpy as np

class HybridDEWithAdaptiveMutation:
    def __init__(self, budget, dim, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.f = f
        self.cr = cr

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        def mutate_population(population, target_index):
            candidates = [idx for idx in range(self.budget) if idx != target_index]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            mutant = population[a] + self.f * (population[b] - population[c])
            crossover_points = np.random.rand(self.dim) < self.cr
            trial = np.where(crossover_points, mutant, population[target_index])
            trial = np.clip(trial, -5.0, 5.0)
            return trial

        population = initialize_population()
        for _ in range(self.budget):
            new_population = np.copy(population)
            for i in range(self.budget):
                trial = mutate_population(population, i)
                if func(trial) < func(population[i]):
                    new_population[i] = trial
            population = new_population
        best_index = np.argmin([func(individual) for individual in population])
        return population[best_index]