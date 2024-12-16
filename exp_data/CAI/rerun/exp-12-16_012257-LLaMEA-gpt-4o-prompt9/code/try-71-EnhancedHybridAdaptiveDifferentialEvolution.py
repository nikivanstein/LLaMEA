import numpy as np

class EnhancedHybridAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = min(max(4 * dim, 20), budget // 2)
        self.F = 0.5
        self.CR = 0.9
        self.population = None
        self.best_individual = None
        self.best_value = float('inf')
        self.base_F = 0.5
        self.base_CR = 0.9
        self.dynamic_shrink_factor = 0.95
        self.progress_threshold = 0.01

    def initialize_population(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
    
    def mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        adaptive_F = self.base_F + (np.random.rand() - 0.5) * 0.1
        mutant = self.population[r1] + adaptive_F * (self.population[r2] - self.population[r3])
        # Added more diversity by including the best individual
        if np.random.rand() < 0.1:
            mutant += self.best_individual - self.population[idx]
        return np.clip(mutant, self.lb, self.ub)
    
    def crossover(self, target, mutant):
        adaptive_CR = self.base_CR + (np.random.rand() - 0.5) * 0.05
        cross_points = np.random.rand(self.dim) < adaptive_CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
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
                    self.base_F = min(0.9, self.base_F * 1.1)
                    self.base_CR = max(0.9, self.base_CR - 0.05)
                    self.population_size = max(10, int(self.population_size * self.dynamic_shrink_factor))
                    self.population = self.population[:self.population_size]
                    stagnation_counter = 0
            else:
                stagnation_counter = 0
            prev_best_value = self.best_value

        return self.best_individual, self.best_value