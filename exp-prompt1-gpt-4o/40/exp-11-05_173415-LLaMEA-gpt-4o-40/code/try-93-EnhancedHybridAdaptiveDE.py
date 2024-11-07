import numpy as np

class EnhancedHybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 8 * dim
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9  # Initial crossover rate
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.elite_ratio = 0.2
        self.elite_count = max(1, int(self.pop_size * self.elite_ratio))
        self.sigma = 0.3
        self.success_memory = []

    def mutate(self, idx):
        elite_indices = np.argsort(self.fitness)[:self.elite_count]
        idxs = np.random.choice(elite_indices, 2, replace=False)
        a, b = self.population[idxs]
        c = self.population[np.random.choice(np.delete(np.arange(self.pop_size), np.concatenate(([idx], idxs))))]
        return a + self.F * (b - c)

    def orthogonal_crossover(self, target, mutant):
        trial = np.copy(target)
        indices = np.random.permutation(self.dim)
        for j in range(self.dim):
            if np.random.rand() < self.CR or j == self.dim - 1:
                trial[indices[j]] = mutant[indices[j]]
        return trial

    def adapt_parameters(self):
        if len(self.success_memory) > 5:
            success_rate = sum(self.success_memory[-5:]) / 5.0
            self.F = np.clip(self.F * (1 + (0.3 if success_rate > 0.3 else -0.1)), 0.1, 0.9)
            self.CR = np.clip(self.CR * (1 + (0.3 if success_rate > 0.3 else -0.1)), 0.1, 1.0)
            self.sigma = np.clip(self.sigma * (1 + (0.15 if success_rate > 0.3 else -0.05)), 0.1, 1.0)

    def __call__(self, func):
        evaluations = 0
        self.fitness = np.array([func(ind) for ind in self.population])
        evaluations += self.pop_size
        
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                
                if np.random.rand() < 0.5:
                    mutant_vec = self.mutate(i)
                else:
                    mutant_vec = self.covariance_mutation(i)
                
                mutant_vec = np.clip(mutant_vec, self.lower_bound, self.upper_bound)
                trial_vec = self.orthogonal_crossover(self.population[i], mutant_vec)
                
                trial_fitness = func(trial_vec)
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vec
                    self.fitness[i] = trial_fitness
                    self.success_memory.append(1)
                else:
                    self.success_memory.append(0)
                evaluations += 1

            self.adapt_parameters()
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]