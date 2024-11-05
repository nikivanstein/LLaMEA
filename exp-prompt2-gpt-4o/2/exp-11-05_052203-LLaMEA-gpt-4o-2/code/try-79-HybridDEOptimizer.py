import numpy as np

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        
        self.population_size = max(5, 4 * self.dim)
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        
        self.F = 0.8  # mutation factor
        self.CR_initial = 0.9  # initial crossover probability
        self.eval_count = 0
        
    def _evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.eval_count += 1
        
    def _mutate(self, idx):
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        dynamic_balance = 1 + np.sqrt((self.budget - self.eval_count) / self.budget)
        adaptive_F = self.F * dynamic_balance * np.random.uniform(0.5, 1.5)
        mutant = self.population[a] + adaptive_F * (self.population[b] - self.population[c])
        return np.clip(mutant, self.lb, self.ub)

    def _crossover(self, target, mutant):
        CR = self.CR_initial * (1.0 - self.eval_count / self.budget)  # dynamic CR adjustment
        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        
        adaptive_factor = np.random.uniform(0.5, 1.5)
        trial = np.clip(adaptive_factor * trial + (1 - adaptive_factor) * target, self.lb, self.ub)

        return trial

    def _select(self, idx, trial_vector, trial_fitness):
        if trial_fitness < self.fitness[idx]:
            self.population[idx] = trial_vector
            self.fitness[idx] = trial_fitness

    def _adaptive_population(self):
        if self.eval_count < self.budget // 3:
            self.population_size = int(self.population_size * 1.5)
        elif 2 * self.budget // 3 < self.eval_count:
            self.population_size = max(5, self.population_size // 2)
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        
    def __call__(self, func):
        self._evaluate_population(func)
        while self.eval_count < self.budget:
            self._adaptive_population()
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                mutant = self._mutate(i)
                trial_vector = self._crossover(self.population[i], mutant)
                trial_fitness = func(trial_vector)
                self.eval_count += 1
                self._select(i, trial_vector, trial_fitness)
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]