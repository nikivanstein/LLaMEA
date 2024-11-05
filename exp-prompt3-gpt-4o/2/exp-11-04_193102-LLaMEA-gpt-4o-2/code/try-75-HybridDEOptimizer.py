import numpy as np

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = max(10, dim * 5)  # Initial population size
        self.pop_size = self.initial_pop_size
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9  # Initial crossover rate
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)

    def __call__(self, func):
        func_calls = 0
        initial_fitness = func(self.population[0])
        while func_calls < self.budget:
            progress = (initial_fitness - np.min(self.fitness)) / (10 * initial_fitness)
            self.pop_size = int(self.initial_pop_size * (1 - progress) + 1)
            self.population = self.population[:self.pop_size]
            self.fitness = self.fitness[:self.pop_size]
            for i in range(self.pop_size):
                indices = np.random.choice([j for j in range(self.pop_size) if j != i], 3, replace=False)
                a, b, c = self.population[indices]
                dynamic_p = 0.02 + progress
                if np.random.rand() < dynamic_p:
                    mutant = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                else:
                    mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])
                
                trial_fitness = func(trial)
                func_calls += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                
                self.F = np.clip(self.F * (1 + 0.01 * (np.random.rand() - 0.5)), 0.1, 1.0)
                self.CR = np.clip(self.CR * (1 + 0.01 * (np.random.rand() - 0.5)), 0.1, 1.0)

                if func_calls >= self.budget:
                    break
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]