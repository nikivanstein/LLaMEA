import numpy as np

class ImprovedBCDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.mutation_rate = 0.8
        self.crossover_rate = 0.9

    def boundary_handling(self, x):
        return np.clip(x, self.lb, self.ub)

    def adjust_population_size(self, population, fitness):
        return population[:self.budget], fitness[:self.budget]

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for _ in range(self.budget):
            idx = np.argsort(fitness)
            best = population[idx[0]]

            mutant = self.boundary_handling(best + self.mutation_rate * (population[idx[1]] - population[idx[2]]))
            trial = np.where(np.random.uniform(0, 1, self.dim) < self.crossover_rate, mutant, population[idx[0]])

            trial_fitness = func(trial)
            if trial_fitness < fitness[idx[0]]:
                population[idx[0]] = trial
                fitness[idx[0]] = trial_fitness

            population, fitness = self.adjust_population_size(population, fitness)

        return population[idx[0]]