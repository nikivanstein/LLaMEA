import numpy as np

class AdaptiveQuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = min(max(5 * dim, 20), budget // 2)
        self.base_F = 0.6
        self.base_CR = 0.8
        self.dynamic_shrink_factor = 0.93
        self.progress_threshold = 0.01
        self.population = None
        self.best_individual = None
        self.best_value = float('inf')

    def initialize_population(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.best_individual = self.population[np.random.randint(self.population_size)]

    def mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        r1, r2 = np.random.choice(indices, 2, replace=False)
        adaptive_F = self.base_F * (1 + np.log10(1 + self.budget / (10 * self.dim)))
        quantum_factor = np.random.normal(0, 1, self.dim)
        mutant = self.population[r1] + adaptive_F * (self.best_individual - self.population[r2]) + quantum_factor
        return np.clip(mutant, self.lb, self.ub)

    def crossover(self, target, mutant):
        adaptive_CR = self.base_CR * (1 - np.exp(-0.1 * self.budget / self.population_size))
        cross_points = np.random.rand(self.dim) < adaptive_CR
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, target_idx, trial, func):
        trial_value = func(trial)
        if trial_value < self.best_value:
            self.best_value = trial_value
            self.best_individual = trial
        if trial_value < func(self.population[target_idx]):
            self.population[target_idx] = trial

    def __call__(self, func):
        self.initialize_population()
        evals = 0
        stagnation_counter = 0
        prev_best_value = self.best_value
        while evals < self.budget:
            for idx in range(self.population_size):
                if evals >= self.budget:
                    break
                target = self.population[idx]
                mutant = self.mutate(idx)
                trial = self.crossover(target, mutant)
                self.select(idx, trial, func)
                evals += 1

            if self.best_value >= prev_best_value - self.progress_threshold:
                stagnation_counter += 1
                if stagnation_counter > 5:
                    self.base_F = min(0.9, self.base_F * 1.15)
                    self.base_CR = max(0.8, self.base_CR - 0.05)
                    self.population_size = max(15, int(self.population_size * self.dynamic_shrink_factor))
                    self.population = self.population[:self.population_size]
                    stagnation_counter = 0
            else:
                stagnation_counter = 0
            prev_best_value = self.best_value

        return self.best_individual, self.best_value