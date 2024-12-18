import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.population = None
        self.best_individual = None
        self.best_value = float('inf')
        self.base_F = 0.5  # New parameter for adaptive F
    
    def initialize_population(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
    
    def mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        adaptive_F = self.base_F + (np.random.rand() - 0.5) * 0.1  # Adaptive F
        mutant = self.population[r1] + adaptive_F * (self.population[r2] - self.population[r3])
        return np.clip(mutant, self.lb, self.ub)
    
    def adaptive_crossover_rate(self):
        diversity = np.mean(np.std(self.population, axis=0))
        return max(0.3, min(1.0, 1.0 - diversity))  # Adaptive CR based on diversity
    
    def crossover(self, target, mutant):
        self.CR = self.adaptive_crossover_rate()  # Update CR dynamically
        cross_points = np.random.rand(self.dim) < self.CR
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
        while evals < self.budget:
            for idx in range(self.population_size):
                if evals >= self.budget:
                    break
                target = self.population[idx]
                mutant = self.mutate(idx)
                trial = self.crossover(target, mutant)
                self.select(idx, trial, func)
                evals += 1
        return self.best_individual, self.best_value