import numpy as np

class ImprovedHybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_pop_size = 8 * dim
        self.F = 0.7
        self.CR = 0.9
        self.learning_rate = 0.1
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_pop_size, self.dim))
        self.fitness = np.full(self.initial_pop_size, np.inf)
        self.elite_ratio = 0.25
        self.sigma = 0.2
        self.min_pop_size = 4 * dim
        self.max_pop_size = 10 * dim

    def mutate(self, idx):
        elite_count = max(1, int(len(self.population) * self.elite_ratio))
        elite_indices = np.argsort(self.fitness)[:elite_count]
        idxs = np.random.choice(elite_indices, 2, replace=False)
        a, b = self.population[idxs]
        c = self.population[np.random.choice(np.delete(np.arange(len(self.population)), np.concatenate(([idx], idxs))))]
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
        adapted_cov = cov + self.learning_rate * np.eye(self.dim)
        return np.random.multivariate_normal(mean, self.sigma * adapted_cov)

    def resize_population(self):
        performance = np.var(self.fitness)
        if performance < 0.05:
            new_size = max(self.min_pop_size, len(self.population) - self.dim)
        else:
            new_size = min(self.max_pop_size, len(self.population) + self.dim)
        self.population = self.population[np.argsort(self.fitness)[:new_size]]
        self.fitness = self.fitness[np.argsort(self.fitness)[:new_size]]

    def __call__(self, func):
        evaluations = 0
        self.fitness = np.array([func(ind) for ind in self.population])
        evaluations += len(self.population)
        
        while evaluations < self.budget:
            self.resize_population()
            for i in range(len(self.population)):
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

            success_rate = np.sum(self.fitness < np.median(self.fitness)) / len(self.population)
            self.F = np.clip(self.F * (1 + (0.3 if success_rate > 0.3 else -0.15)), 0.1, 1.0)
            self.CR = np.clip(self.CR * (1 + (0.3 if success_rate > 0.3 else -0.15)), 0.1, 1.0)
            self.sigma = np.clip(self.sigma * (1 + (0.15 if success_rate > 0.3 else -0.075)), 0.1, 1.0)
            self.learning_rate = np.clip(self.learning_rate * (1 + (0.05 if success_rate > 0.3 else -0.025)), 0.01, 0.2)
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]