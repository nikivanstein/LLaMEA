import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 8 * dim
        self.F = 0.6
        self.CR = 0.8
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.elite_ratio = 0.2
        self.elite_count = max(1, int(self.pop_size * self.elite_ratio))
        self.global_best = None
        self.global_best_fit = np.inf
        
    def mutate(self, idx):
        elite_indices = np.argsort(self.fitness)[:self.elite_count]
        idxs = np.random.choice(elite_indices, 2, replace=False)
        a, b = self.population[idxs]
        c = self.population[np.random.choice(np.delete(np.arange(self.pop_size), np.concatenate(([idx], idxs))))]
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
    
    def adaptive_scaling(self):
        diversity = np.mean(np.std(self.population, axis=0))
        self.F = np.clip(self.F * (1 + (0.5 if diversity < 0.1 else -0.3)), 0.1, 0.9)
        self.CR = np.clip(self.CR * (1 + (0.5 if diversity < 0.1 else -0.3)), 0.1, 0.9)
    
    def __call__(self, func):
        evaluations = 0
        self.fitness = np.array([func(ind) for ind in self.population])
        evaluations += self.pop_size
        
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                
                mutant_vec = self.mutate(i)
                mutant_vec = np.clip(mutant_vec, self.lower_bound, self.upper_bound)
                trial_vec = self.crossover(self.population[i], mutant_vec)
                
                if self.select(trial_vec, i, func):
                    evaluations += 1
                    if self.fitness[i] < self.global_best_fit:
                        self.global_best = self.population[i]
                        self.global_best_fit = self.fitness[i]
                else:
                    evaluations += 1
            
            self.adaptive_scaling()
            if evaluations % (self.pop_size // 2) == 0:
                exploration = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                new_candidate = np.clip(self.global_best + self.F * (exploration - self.global_best), self.lower_bound, self.upper_bound)
                self.select(new_candidate, np.argmax(self.fitness), func)
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]