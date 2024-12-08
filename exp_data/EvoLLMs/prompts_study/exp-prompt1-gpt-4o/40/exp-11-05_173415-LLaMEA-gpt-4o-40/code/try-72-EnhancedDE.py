import numpy as np

class EnhancedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 8 * dim
        self.min_pop_size = 4 * dim
        self.F = 0.6
        self.CR = 0.8
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.archive = []
        self.archive_size = int(0.5 * self.pop_size)
        
    def mutate(self, idx):
        idxs = np.random.choice(np.delete(np.arange(self.pop_size), idx), 3, replace=False)
        a, b, c = self.population[idxs]
        return a + self.F * (b - c)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def adapt_parameters(self, trial, target_idx, func):
        trial_fitness = func(trial)
        if trial_fitness < self.fitness[target_idx]:
            self.archive.append(self.population[target_idx].copy())
            if len(self.archive) > self.archive_size:
                self.archive.pop(0)
            self.population[target_idx] = trial
            self.fitness[target_idx] = trial_fitness
            return True, trial_fitness
        return False, self.fitness[target_idx]

    def archive_mutation(self, idx):
        if len(self.archive) == 0:
            return self.mutate(idx)
        archive_sample = self.archive[np.random.randint(len(self.archive))]
        idxs = np.random.choice(np.delete(np.arange(self.pop_size), idx), 2, replace=False)
        a, b = self.population[idxs]
        return a + self.F * (b - archive_sample)

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
                    mutant_vec = self.archive_mutation(i)
                
                mutant_vec = np.clip(mutant_vec, self.lower_bound, self.upper_bound)
                trial_vec = self.crossover(self.population[i], mutant_vec)
                
                successful, new_fitness = self.adapt_parameters(trial_vec, i, func)
                evaluations += 1

            success_rate = np.sum(self.fitness < np.median(self.fitness)) / self.pop_size
            self.F = np.clip(self.F * (1 + (0.2 if success_rate > 0.2 else -0.1)), 0.1, 0.9)
            self.CR = np.clip(self.CR * (1 + (0.2 if success_rate > 0.2 else -0.1)), 0.1, 0.9)

            if self.pop_size > self.min_pop_size and np.random.rand() < 0.1:
                self.pop_size = max(self.min_pop_size, int(self.pop_size * 0.9))
                self.population = self.population[:self.pop_size]
                self.fitness = self.fitness[:self.pop_size]
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]