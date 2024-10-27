import numpy as np

class ImprovedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.cr = 0.9
        self.f_min = 0.2
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')

    def mutate(self, target_idx, population, iter_count):
        candidates = [idx for idx in range(self.population_size) if idx != target_idx]
        selected = np.random.choice(candidates, 3, replace=False)
        f_dynamic = max(0.1, self.f_min * (1 - iter_count / self.budget))
        mutant = population[selected[0]] + f_dynamic * (population[selected[1]] - population[selected[2]])
        return mutant

    def crossover(self, target, mutant, iter_count):
        trial = np.copy(target)
        for i in range(self.dim):
            cr_dynamic = max(0.1, self.cr * (1 - iter_count / self.budget))
            if np.random.random() < cr_dynamic:
                trial[i] = mutant[i]
        return trial

    def __call__(self, func):
        for iter_count in range(self.budget):
            for i in range(self.population_size):
                mutant = self.mutate(i, self.population, iter_count)
                trial = self.crossover(self.population[i], mutant, iter_count)
                if func(trial) < func(self.population[i]):
                    self.population[i] = trial
            fitnesses = np.array([func(ind) for ind in self.population])
            best_idx = np.argmin(fitnesses)
            if fitnesses[best_idx] < self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_solution = np.copy(self.population[best_idx])
        return self.best_solution