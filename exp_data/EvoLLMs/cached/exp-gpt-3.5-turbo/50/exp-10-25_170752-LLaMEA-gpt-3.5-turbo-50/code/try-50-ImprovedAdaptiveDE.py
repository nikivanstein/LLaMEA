import numpy as np

class ImprovedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.cr_min = 0.2
        self.cr_max = 0.9
        self.f_min = 0.2
        self.f_max = 0.8
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')

    def mutate(self, target_idx, population, f_val):
        candidates = [idx for idx in range(self.population_size) if idx != target_idx]
        selected = np.random.choice(candidates, 3, replace=False)
        mutant = population[selected[0]] + f_val * (population[selected[1]] - population[selected[2]])
        return mutant

    def crossover(self, target, mutant, cr_val):
        trial = np.copy(target)
        for i in range(self.dim):
            if np.random.random() < cr_val:
                trial[i] = mutant[i]
        return trial

    def adapt_parameters(self, iter_count):
        self.cr = self.cr_min + (self.cr_max - self.cr_min) * (iter_count / self.budget)
        self.f_val = self.f_min + (self.f_max - self.f_min) * (iter_count / self.budget)

    def __call__(self, func):
        for iter_count in range(self.budget):
            self.adapt_parameters(iter_count)
            for i in range(self.population_size):
                mutant = self.mutate(i, self.population, self.f_val)
                trial = self.crossover(self.population[i], mutant, self.cr)
                if func(trial) < func(self.population[i]):
                    self.population[i] = trial
            fitnesses = np.array([func(ind) for ind in self.population])
            best_idx = np.argmin(fitnesses)
            if fitnesses[best_idx] < self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_solution = np.copy(self.population[best_idx])
        return self.best_solution