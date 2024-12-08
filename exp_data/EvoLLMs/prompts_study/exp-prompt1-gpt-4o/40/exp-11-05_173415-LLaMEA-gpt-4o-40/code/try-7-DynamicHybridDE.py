import numpy as np

class DynamicHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_pop_size = 10 * dim
        self.pop_size = self.initial_pop_size
        self.F = 0.5
        self.CR = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.adapt_factor = 0.1
        self.local_search_radius = 0.1
        
    def mutate(self, idx):
        idxs = np.random.choice(np.delete(np.arange(self.pop_size), [idx]), 3, replace=False)
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
    
    def adapt_population_size(self):
        improvement_rate = np.mean(self.fitness < np.median(self.fitness))
        if improvement_rate < 0.2:
            self.pop_size = max(int(self.pop_size * (1 - self.adapt_factor)), 4)
        else:
            self.pop_size = min(int(self.pop_size * (1 + self.adapt_factor)), self.initial_pop_size)
        self.population = self.population[:self.pop_size]
        self.fitness = self.fitness[:self.pop_size]
    
    def local_search(self, best_idx, func):
        best = self.population[best_idx]
        perturbation = np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
        local_candidate = np.clip(best + perturbation, self.lower_bound, self.upper_bound)
        candidate_fitness = func(local_candidate)
        if candidate_fitness < self.fitness[best_idx]:
            self.population[best_idx] = local_candidate
            self.fitness[best_idx] = candidate_fitness
    
    def __call__(self, func):
        evaluations = 0
        self.fitness = np.array([func(ind) for ind in self.population])
        evaluations += self.pop_size
        
        while evaluations < self.budget:
            improved = False
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                
                mutant_vec = self.mutate(i)
                mutant_vec = np.clip(mutant_vec, self.lower_bound, self.upper_bound)
                trial_vec = self.crossover(self.population[i], mutant_vec)
                
                if self.select(trial_vec, i, func):
                    evaluations += 1
                    improved = True
                else:
                    evaluations += 1
                
                if evaluations >= self.budget:
                    break
            
            self.adapt_population_size()
            
            if improved:
                best_idx = np.argmin(self.fitness)
                self.local_search(best_idx, func)
                evaluations += 1
            
            if evaluations % (self.pop_size * 10) == 0:
                success_rate = np.sum(self.fitness < np.median(self.fitness)) / self.pop_size
                self.F = np.clip(self.F * (1 + (0.2 if success_rate > 0.2 else -0.1)), 0.1, 0.9)
                self.CR = np.clip(self.CR * (1 + (0.2 if success_rate > 0.2 else -0.1)), 0.1, 0.9)
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]