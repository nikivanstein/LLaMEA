import numpy as np

class EnhancedAdaptiveEliteDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_pop_size = 8 * dim
        self.pop_size = self.initial_pop_size
        self.F = 0.6
        self.CR = 0.8
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.elite_ratio = 0.2
        self.elite_count = max(1, int(self.pop_size * self.elite_ratio))
        self.feedback_threshold = 0.1
    
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
    
    def update_parameters(self, success_rate):
        self.F = np.clip(self.F * (1 + (0.2 if success_rate > 0.3 else -0.1)), 0.2, 0.9)
        self.CR = np.clip(self.CR * (1 + (0.2 if success_rate > 0.3 else -0.1)), 0.2, 0.9)
        if success_rate < self.feedback_threshold:
            self.pop_size = min(self.pop_size + 1, self.initial_pop_size * 2)
            new_individuals = np.random.uniform(self.lower_bound, self.upper_bound, (1, self.dim))
            self.population = np.vstack([self.population, new_individuals])
            self.fitness = np.append(self.fitness, np.inf)

    def __call__(self, func):
        evaluations = 0
        self.fitness = np.array([func(ind) for ind in self.population])
        evaluations += self.pop_size
        
        while evaluations < self.budget:
            successful_trials = 0
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                mutant_vec = self.mutate(i)
                mutant_vec = np.clip(mutant_vec, self.lower_bound, self.upper_bound)
                trial_vec = self.crossover(self.population[i], mutant_vec)
                
                if self.select(trial_vec, i, func):
                    successful_trials += 1
                evaluations += 1

            success_rate = successful_trials / self.pop_size
            self.update_parameters(success_rate)
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]