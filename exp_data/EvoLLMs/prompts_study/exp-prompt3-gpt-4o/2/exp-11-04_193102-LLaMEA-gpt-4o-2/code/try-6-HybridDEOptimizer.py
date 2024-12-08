import numpy as np

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(10, dim * 5)  # Population size
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9  # Initial crossover rate
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)

    def __call__(self, func):
        func_calls = 0
        dynamic_dim = self.dim
        while func_calls < self.budget:
            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice([j for j in range(self.pop_size) if j != i], 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                # Crossover
                cross_points = np.random.rand(dynamic_dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dynamic_dim)] = True
                trial = np.where(cross_points, mutant, self.population[i][:dynamic_dim])
                
                # Selection
                trial_fitness = func(trial)
                func_calls += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i][:dynamic_dim] = trial
                    self.fitness[i] = trial_fitness
                
                # Adaptation of F and CR
                self.F = np.clip(self.F * (1 + 0.01 * (np.random.rand() - 0.5)), 0.1, 1.0)
                self.CR = np.clip(self.CR * (1 + 0.01 * (np.random.rand() - 0.5)), 0.1, 1.0)

                # Dynamic adjustment of dimensions
                if func_calls % (self.budget // 10) == 0:
                    dynamic_dim = max(1, dynamic_dim - 1)

                if func_calls >= self.budget:
                    break
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]