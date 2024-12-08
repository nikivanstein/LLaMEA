import numpy as np

class DynamicDEAlgorithm:
    def __init__(self, budget, dim, population_size=50, cr=0.9, f=0.8):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.cr = cr
        self.f = f

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def mutate(self, population, target_idx):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        mutant = population[idxs[0]] + self.f * (population[idxs[1]] - population[idxs[2]])
        return mutant

    def crossover(self, target, mutant):
        trial = np.copy(target)
        rand_dims = np.random.rand(self.dim) < self.cr
        trial[rand_dims] = mutant[rand_dims]
        return trial

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(individual) for individual in population])

        for _ in range(self.budget):
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                mutant = self.mutate(population, i)
                trial = self.crossover(population[i], mutant)
                new_population[i] = trial if func(trial) < fitness[i] else population[i]
            population = new_population
            fitness = np.array([func(individual) for individual in population])

        return population[np.argmin(fitness)]