import numpy as np

class SelfAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.cr_min = 0.1
        self.f_min = 0.2
        self.cr_range = [0.1, 0.9]  # Range for self-adaptive crossover rate
        self.f_range = [0.1, 0.9]   # Range for self-adaptive mutation factor
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.cr = np.random.uniform(self.cr_range[0], self.cr_range[1])
        self.f = np.random.uniform(self.f_range[0], self.f_range[1])

    def mutate(self, target_idx, population):
        candidates = [idx for idx in range(self.population_size) if idx != target_idx]
        selected = np.random.choice(candidates, 3, replace=False)
        mutant = population[selected[0]] + self.f * (population[selected[1]] - population[selected[2]])
        return mutant

    def crossover(self, target, mutant):
        trial = np.copy(target)
        for i in range(self.dim):
            if np.random.random() < self.cr:
                trial[i] = mutant[i]
        return trial

    def self_adapt_parameters(self):
        self.cr = np.clip(self.cr + np.random.normal(0, 0.1), self.cr_range[0], self.cr_range[1])
        self.f = np.clip(self.f + np.random.normal(0, 0.1), self.f_range[0], self.f_range[1])

    def __call__(self, func):
        for iter_count in range(self.budget):
            self.self_adapt_parameters()
            for i in range(self.population_size):
                mutant = self.mutate(i, self.population)
                trial = self.crossover(self.population[i], mutant)
                if func(trial) < func(self.population[i]):
                    self.population[i] = trial
            fitnesses = np.array([func(ind) for ind in self.population])
            best_idx = np.argmin(fitnesses)
            if fitnesses[best_idx] < self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_solution = np.copy(self.population[best_idx])
        return self.best_solution