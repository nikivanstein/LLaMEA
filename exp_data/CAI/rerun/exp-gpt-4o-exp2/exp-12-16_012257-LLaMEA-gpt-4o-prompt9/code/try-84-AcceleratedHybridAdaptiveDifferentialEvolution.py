import numpy as np

class AcceleratedHybridAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = min(max(5 * dim, 30), budget // 2)  # Adjusted population size
        self.base_F = 0.5
        self.base_CR = 0.9
        self.dynamic_shrink_factor = 0.9  # More aggressive shrink factor
        self.progress_threshold = 0.001  # Reduced threshold for quicker response
        self.population = None
        self.best_individual = None
        self.best_value = float('inf')
        self.strategy_switch_threshold = 0.25

    def initialize_population(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
    
    def mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        dynamic_F = self.base_F + (np.random.rand() - 0.5) * 0.3  # Increased F variability
        mutant = self.population[r1] + dynamic_F * (self.population[r2] - self.population[r3])
        if np.random.rand() < 0.2:  # Increased probability to use the best individual
            mutant += self.best_individual - self.population[idx]
        return np.clip(mutant, self.lb, self.ub)
    
    def crossover(self, target, mutant):
        adaptive_CR = self.base_CR + (np.random.rand() - 0.5) * 0.1  # Broader CR adaptation
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
        adapt_mutation = True

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
                if stagnation_counter > 3:  # Faster response to stagnation
                    if adapt_mutation:
                        self.base_F = min(0.9, self.base_F * 1.2)
                        self.base_CR = max(0.8, self.base_CR - 0.1)
                    else:
                        self.population_size = max(10, int(self.population_size * self.dynamic_shrink_factor))
                        self.population = self.population[:self.population_size]
                    adapt_mutation = not adapt_mutation
                    stagnation_counter = 0
            else:
                stagnation_counter = 0
            prev_best_value = self.best_value

        return self.best_individual, self.best_value