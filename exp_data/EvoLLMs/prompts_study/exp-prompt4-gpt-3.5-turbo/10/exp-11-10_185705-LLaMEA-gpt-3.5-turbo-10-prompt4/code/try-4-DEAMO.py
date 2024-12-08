import numpy as np

class DEAMO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.cr = 0.5
        self.f = 0.5
        self.mutation_strategies = np.random.uniform(0, 1, (self.population_size, self.dim))

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evals = self.population_size

        while evals < self.budget:
            new_population = np.zeros_like(population)
            for i in range(self.population_size):
                candidates = np.random.choice(population, 3, replace=False)
                mutant = candidates[0] + self.mutation_strategies[i] * (candidates[1] - candidates[2])
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, population[i])
                new_fitness = func(trial)
                evals += 1
                if new_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = new_fitness
            self.adapt_mutation_strategies(fitness)
        best_idx = np.argmin(fitness)
        return population[best_idx]

    def adapt_mutation_strategies(self, fitness):
        best_idx = np.argmin(fitness)
        for i in range(self.population_size):
            if i != best_idx:
                self.mutation_strategies[i] = self.mutation_strategies[i] + self.f * (self.mutation_strategies[best_idx] - self.mutation_strategies[i])
                self.mutation_strategies[i] = np.clip(self.mutation_strategies[i], 0, 1)