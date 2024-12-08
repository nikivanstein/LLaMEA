import numpy as np

class RefinedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim  # Increased population size for diversity
        self.F = 0.5  # Adjusted initial mutation factor
        self.CR = 0.9  # Adjusted crossover rate
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.elite_ratio = 0.3  # Adjusted elite ratio
        self.elite_count = max(1, int(self.pop_size * self.elite_ratio))
        self.sigma = 0.2  # Adjusted sigma for covariance mutation

    def mutate(self, idx):
        elite_indices = np.argsort(self.fitness)[:self.elite_count]
        idxs = np.random.choice(elite_indices, 3, replace=False)
        a, b, c = self.population[idxs]
        return a + self.F * (b - c)  # Differential mutation strategy

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

    def covariance_mutation(self):
        mean = np.mean(self.population, axis=0)
        cov = np.cov(self.population, rowvar=False)
        return np.random.multivariate_normal(mean, self.sigma * cov)

    def __call__(self, func):
        evaluations = 0
        self.fitness = np.array([func(ind) for ind in self.population])
        evaluations += self.pop_size
        
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                
                if np.random.rand() < 0.7:  # Adjusted probability for mutation strategy selection
                    mutant_vec = self.mutate(i)
                else:
                    mutant_vec = self.covariance_mutation()
                
                mutant_vec = np.clip(mutant_vec, self.lower_bound, self.upper_bound)
                trial_vec = self.crossover(self.population[i], mutant_vec)
                
                successful, new_fitness = self.adapt_parameters(trial_vec, i, func)
                evaluations += 1

            success_rate = np.sum(self.fitness < np.median(self.fitness)) / self.pop_size
            self.F = np.clip(self.F * (1 + (0.3 if success_rate > 0.3 else -0.1)), 0.1, 0.9)  # Enhanced adaptation
            self.CR = np.clip(self.CR * (1 + (0.3 if success_rate > 0.3 else -0.1)), 0.1, 1.0)
            self.sigma = np.clip(self.sigma * (1 + (0.2 if success_rate > 0.3 else -0.05)), 0.1, 0.5)  # Adjusted range
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]