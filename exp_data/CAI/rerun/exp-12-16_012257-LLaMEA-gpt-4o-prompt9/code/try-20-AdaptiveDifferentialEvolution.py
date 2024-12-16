import numpy as np

class AdaptiveDifferentialEvolution:
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
    
    def initialize_population(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
    
    def mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        dynamic_F = self.base_F * (1 + np.random.rand() * 0.1)  # Modified mutation factor adjustment
        mutant = self.population[r1] + dynamic_F * (self.population[r2] - self.population[r3])
        return np.clip(mutant, self.lb, self.ub)
    
    def crossover(self, target, mutant):
        dynamic_CR = self.base_CR * (1 - np.random.rand() * 0.05)  # Modified crossover rate adjustment
        cross_points = np.random.rand(self.dim) < dynamic_CR
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
        convergence_threshold = 1e-5  # Convergence condition adjustment
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
            
            # Adjust mutation factor and crossover rate dynamically
            if abs(self.best_value - prev_best_value) < convergence_threshold:
                self.base_F = max(0.4, self.base_F * 0.9)
                self.base_CR = min(1.0, self.base_CR * 1.1)
            prev_best_value = self.best_value

        return self.best_individual, self.best_value