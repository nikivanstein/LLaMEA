import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_pop_size = 8 * dim
        self.F = 0.6
        self.CR = 0.8
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_pop_size, self.dim))
        self.fitness = np.full(self.initial_pop_size, np.inf)
        self.elite_ratio = 0.2
        self.elite_count = max(1, int(self.initial_pop_size * self.elite_ratio))
        self.sigma = 0.3
        self.min_pop_size = 4 * dim

    def mutate(self, idx):
        elite_indices = np.argsort(self.fitness)[:self.elite_count]
        idxs = np.random.choice(elite_indices, 2, replace=False)
        a, b = self.population[idxs]
        c = self.population[np.random.choice(np.delete(np.arange(self.initial_pop_size), np.concatenate(([idx], idxs))))]
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

    def covariance_mutation(self, idx):
        mean = np.mean(self.population, axis=0)
        cov = np.cov(self.population, rowvar=False)
        return np.random.multivariate_normal(mean, self.sigma * cov)

    def stochastic_ranking(self):
        indices = np.arange(len(self.fitness))
        np.random.shuffle(indices)
        sorted_indices = sorted(indices, key=lambda idx: (self.fitness[idx], np.random.rand()))
        return sorted_indices[:self.elite_count]

    def __call__(self, func):
        evaluations = 0
        self.fitness = np.array([func(ind) for ind in self.population])
        evaluations += self.initial_pop_size
        
        while evaluations < self.budget:
            current_pop_size = max(self.min_pop_size, int(self.initial_pop_size * (1 - evaluations / self.budget)))
            for i in range(current_pop_size):
                if evaluations >= self.budget:
                    break
                
                if np.random.rand() < 0.5:
                    mutant_vec = self.mutate(i)
                else:
                    mutant_vec = self.covariance_mutation(i)
                
                mutant_vec = np.clip(mutant_vec, self.lower_bound, self.upper_bound)
                trial_vec = self.crossover(self.population[i], mutant_vec)
                
                successful, new_fitness = self.adapt_parameters(trial_vec, i, func)
                evaluations += 1

            elites = self.stochastic_ranking()
            self.F = np.clip(self.F * (1 + (0.2 if len(elites) > 0.2 * current_pop_size else -0.1)), 0.1, 0.9)
            self.CR = np.clip(self.CR * (1 + (0.2 if len(elites) > 0.2 * current_pop_size else -0.1)), 0.1, 0.9)
            self.sigma = np.clip(self.sigma * (1 + (0.1 if len(elites) > 0.2 * current_pop_size else -0.05)), 0.1, 1.0)
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]