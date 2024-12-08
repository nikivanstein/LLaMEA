import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim, population_size=None):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size or 20 * dim
        self.population = np.random.uniform(-5, 5, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.F = 0.5  # initial mutation factor
        self.CR = 0.9  # initial crossover rate
        self.evaluations = 0
        self.success_rate = 0.0  # Track success rate to adjust parameters

    def __call__(self, func):
        best_idx = np.argmin(self.fitness)
        best = self.population[best_idx]
        
        while self.evaluations < self.budget:
            successful_trials = 0  # Track successful trials
            
            if self.evaluations % (self.budget // 10) == 0:
                self.population_size = min(int(self.population_size * 0.9), self.population_size)
            
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), -5, 5)
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, self.population[i])
                
                trial_fitness = func(trial)
                self.evaluations += 1
                
                if trial_fitness < self.fitness[i]:
                    successful_trials += 1  # Increment success count
                    self.fitness[i] = trial_fitness
                    self.population[i] = trial
                    if trial_fitness < self.fitness[best_idx]:
                        best = trial
                        best_idx = i
            
            # Adjust F and CR based on successful trials
            self.success_rate = successful_trials / self.population_size
            self.F = np.clip(self.F + 0.05 * (np.random.rand() - 0.5) * (1 if self.success_rate > 0.2 else -1), 0.4, 0.9)
            self.CR = np.clip(self.CR + 0.05 * (np.random.rand() - 0.5) * (1 if self.success_rate > 0.2 else -1), 0.8, 1.0)

        return best