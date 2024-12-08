import numpy as np

class ImprovedAdaptiveEliteDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 8 * dim
        self.F_base = 0.6
        self.CR_base = 0.8
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.elite_ratio = 0.2
        self.elite_count = max(1, int(self.pop_size * self.elite_ratio))
        
    def mutate(self, idx, strategy="rand"):
        if strategy == "best":
            best_idx = np.argmin(self.fitness)
            a = self.population[best_idx]
        else:  # "rand"
            elite_indices = np.argsort(self.fitness)[:self.elite_count]
            idxs = np.random.choice(elite_indices, 2, replace=False)
            a, b = self.population[idxs]
        c = self.population[np.random.choice(np.delete(np.arange(self.pop_size), [idx]))]
        return a + self.F_base * (b - c)
    
    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR_base
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
        
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                
                strategy = "best" if np.random.rand() < 0.5 else "rand"
                mutant_vec = self.mutate(i, strategy)
                mutant_vec = np.clip(mutant_vec, self.lower_bound, self.upper_bound)
                trial_vec = self.crossover(self.population[i], mutant_vec)
                
                if self.select(trial_vec, i, func):
                    evaluations += 1
                else:
                    evaluations += 1

            elite_improvement = np.sum(self.fitness < np.median(self.fitness)) / self.pop_size
            self.F_base = np.clip(self.F_base * (1 + (0.3 if elite_improvement > 0.2 else -0.15)), 0.1, 1.0)
            self.CR_base = np.clip(self.CR_base * (1 + (0.3 if elite_improvement > 0.2 else -0.15)), 0.1, 1.0)
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]