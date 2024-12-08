import numpy as np

class AdaptiveHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.elite_ratio = 0.2
        self.elite_count = max(1, int(self.pop_size * self.elite_ratio))
        self.strategy_change_interval = 5
        
    def mutate(self, idx):
        elite_indices = np.argsort(self.fitness)[:self.elite_count]
        idxs = np.random.choice(elite_indices, 3, replace=False)
        a, b, c = self.population[idxs]
        return a + self.F * (b - c)
    
    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial
    
    def select(self, trial, target_idx, func):
        trial_fitness = func(trial)
        if trial_fitness < self.fitness[target_idx]:
            self.population[target_idx] = trial
            self.fitness[target_idx] = trial_fitness
            return True
        return False
    
    def __call__(self, func):
        evaluations = 0
        self.fitness = np.array([func(ind) for ind in self.population])
        evaluations += self.pop_size
        
        strategy_switch = 0
        
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                
                mutant_vec = self.mutate(i)
                mutant_vec = np.clip(mutant_vec, self.lower_bound, self.upper_bound)
                trial_vec = self.crossover(self.population[i], mutant_vec)
                
                if self.select(trial_vec, i, func):
                    evaluations += 1
                else:
                    evaluations += 1
                
            if evaluations // self.pop_size > strategy_switch:
                strategy_switch += 1
                if strategy_switch % self.strategy_change_interval == 0:
                    self.CR = np.clip(self.CR * np.random.uniform(0.8, 1.2), 0.1, 0.9)
            
            self.F = np.clip(self.F * (1 + 0.2 * np.random.uniform(-1, 1)), 0.1, 1.0)
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]