import numpy as np

class EnhancedHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 8 * dim
        self.F = 0.7
        self.CR = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.elite_archive = np.zeros((int(self.pop_size * 0.1), self.dim))
        self.sigma = 0.3

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
            self.population[target_idx] = trial
            self.fitness[target_idx] = trial_fitness
            return True, trial_fitness
        return False, self.fitness[target_idx]

    def archive_elite(self):
        elite_indices = np.argsort(self.fitness)[:self.elite_archive.shape[0]]
        self.elite_archive = self.population[elite_indices]

    def covariance_mutation(self):
        if len(self.elite_archive) > 0:
            archive_mean = np.mean(self.elite_archive, axis=0)
            cov = np.cov(self.elite_archive, rowvar=False)
            return np.random.multivariate_normal(archive_mean, self.sigma * cov)
        return None

    def __call__(self, func):
        evaluations = 0
        self.fitness = np.array([func(ind) for ind in self.population])
        evaluations += self.pop_size
        
        while evaluations < self.budget:
            self.archive_elite()
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                if np.random.rand() < 0.6:
                    mutant_vec = self.mutate(i)
                else:
                    cov_mutation = self.covariance_mutation()
                    if cov_mutation is not None:
                        mutant_vec = cov_mutation
                    else:
                        mutant_vec = self.mutate(i)
                
                mutant_vec = np.clip(mutant_vec, self.lower_bound, self.upper_bound)
                trial_vec = self.crossover(self.population[i], mutant_vec)
                
                successful, new_fitness = self.adapt_parameters(trial_vec, i, func)
                evaluations += 1

            success_rate = np.sum(self.fitness < np.median(self.fitness)) / self.pop_size
            self.F = np.clip(self.F * (1 + (0.1 if success_rate > 0.2 else -0.05)), 0.4, 0.9)
            self.CR = np.clip(self.CR * (1 + (0.1 if success_rate > 0.2 else -0.05)), 0.4, 0.9)
            self.sigma = np.clip(self.sigma * (1 + (0.05 if success_rate > 0.2 else -0.02)), 0.1, 1.0)
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]